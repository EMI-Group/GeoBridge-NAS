"""DDAD split-file dataset loader for retraining."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .sparse_sampler import simulate_sparse_depth

logger = logging.getLogger(__name__)


class DDADDataset(Dataset):
    """
    DDAD split loader compatible with iDisc-style split files.

    Entries are expected to be one of:
        <rgb> <depth>
        <rgb> <depth> <sparse_depth>
        <rgb> <depth> <depth_scale>
        <rgb> <depth> <sparse_depth> <depth_scale>

    Relative RGB paths are resolved under ``root``. Relative depth and sparse
    paths are resolved under ``gt_root`` when provided, otherwise under ``root``.
    Depth PNG values are divided by ``depth_scale`` to convert to metres.

    Preprocessing matches the PixelFormer DDAD protocol shipped in ``ddad.zip``:
    the **full** frame is resized to ``img_size`` (no sky/hood pre-crop) and
    evaluation uses **no** garg/eigen crop with a ``[1e-3, 200] m`` depth range,
    so keep ``min_depth`` / ``max_depth`` at ``1e-3`` / ``200`` so ``valid_mask``
    matches the trainer's eval mask.

    zip evaluates at ``608x960``. zip's *training* resizes the full frame to
    ``608x960`` then random-crops to ``480x640``: replicate that by building the
    train split with ``img_size=(480, 640)`` and ``train_resize=(608, 960)``,
    and the val split with ``img_size=(608, 960)`` (no ``train_resize``). The
    legacy iDisc fixed crop is still available via ``crop_box`` but is off by
    default.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEFAULT_CROP_BOX = (180, 8, 1050, 1928)
    DEFAULT_PRE_RESIZE_SIZE = (870, 1920)

    def __init__(
        self,
        root: str,
        filenames_file: str,
        split: str = "train",
        gt_root: Optional[str] = None,
        intrinsics_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        img_size: Tuple[int, int] = (608, 960),
        depth_scale: float = 256.0,
        min_depth: float = 1e-3,
        max_depth: float = 200.0,
        sparse_ratio: float = 0.0,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        camera09_bottom_mask: int = 0,
        train_resize: Optional[Tuple[int, int]] = None,
        rotate_degree: float = 2.5,
        cutdepth_prob: float = 0.25,
    ):
        super().__init__()
        self.root = Path(root)
        self.gt_root = Path(gt_root) if gt_root else self.root
        self.filenames_file = Path(filenames_file)
        self.split = split
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sparse_ratio = sparse_ratio
        self.crop_box = crop_box
        self.camera09_bottom_mask = camera09_bottom_mask
        self.train_resize = train_resize
        self.rotate_degree = rotate_degree
        self.cutdepth_prob = cutdepth_prob
        self.transform = transform
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        self.intrinsics = self._load_intrinsics(intrinsics_file)

        if not self.filenames_file.exists():
            raise FileNotFoundError(self.filenames_file)
        with open(self.filenames_file) as f:
            samples: List[List[str]] = [
                line.strip().split()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if split != "train":
            samples = [
                fields for fields in samples
                if len(fields) >= 2 and fields[1].lower() != "none"
            ]
        self.samples = samples[:max_samples] if max_samples is not None else samples

        # Make train-time augmentation state visible at startup so it can never
        # be silently stale again (a running process never reloads this file).
        if self.split == "train" and os.environ.get("LOCAL_RANK", "0") == "0":
            logger.info(
                "DDAD train aug: flip=0.5, photometric=0.5, rotate_degree=%.2f, "
                "cutdepth_prob=%.2f, train_resize=%s -> crop=%s",
                self.rotate_degree, self.cutdepth_prob,
                self.train_resize, self.img_size,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fields = self.samples[idx]
        if len(fields) < 2:
            raise ValueError(f"Invalid DDAD split entry: {' '.join(fields)}")

        img_rel = fields[0]
        img_path = self._resolve(self.root, img_rel)
        depth_path = self._resolve(self.gt_root, fields[1])
        sparse_path = self._parse_sparse_depth_path(fields)
        sample_scale = self._parse_depth_scale(fields)

        image = Image.open(img_path).convert("RGB")
        depth = self._load_depth(depth_path, sample_scale)
        sparse = self._load_depth(sparse_path, sample_scale) if sparse_path is not None else None
        K = self._intrinsics_for(img_rel)

        image, depth, sparse, K = self._crop_resize(image, depth, sparse, K)
        if self.split == "train":
            image_tensor, depth, sparse, K = self._preprocess_train(image, depth, sparse, K)
        else:
            image_tensor = self._preprocess_eval_image(image)

        valid_mask = (
            torch.isfinite(depth)
            & (depth > self.min_depth)
            & (depth < self.max_depth)
        )
        valid_mask = valid_mask & self._camera_mask(img_rel, depth.shape[-2], depth.shape[-1])
        depth = torch.clamp(depth.float(), min=0.0, max=self.max_depth)

        result = {
            "image": image_tensor,
            "depth": depth,
            "valid_mask": valid_mask,
            "K": K,
            "idx": idx,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }
        if sparse is not None:
            result["sparse_depth"] = torch.clamp(sparse.float(), min=0.0, max=self.max_depth)
        elif self.sparse_ratio > 0:
            result["sparse_depth"] = simulate_sparse_depth(depth, self.sparse_ratio)
        return result

    def _load_intrinsics(self, intrinsics_file: Optional[str]) -> Dict[str, Tensor]:
        candidates = []
        if intrinsics_file is not None:
            candidates.append(Path(intrinsics_file))
        candidates.append(self.root / "ddad_intrinsics.json")

        for path in candidates:
            if path.exists():
                with open(path) as f:
                    raw = json.load(f)
                return {
                    key: torch.tensor(value, dtype=torch.float32).view(3, 3)
                    for key, value in raw.items()
                }
        return {}

    def _resolve(self, root: Path, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else root / p

    def _parse_sparse_depth_path(self, fields: List[str]) -> Optional[Path]:
        if len(fields) >= 3 and not self._looks_like_number(fields[2]):
            return self._resolve(self.gt_root, fields[2])
        return None

    def _parse_depth_scale(self, fields: List[str]) -> float:
        for value in reversed(fields[2:]):
            if self._looks_like_number(value):
                return float(value)
        return self.depth_scale

    @staticmethod
    def _looks_like_number(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _intrinsics_for(self, img_rel: str) -> Tensor:
        K = self.intrinsics.get(img_rel)
        return K.clone() if K is not None else torch.eye(3, dtype=torch.float32)

    def _load_depth(self, path: Path, depth_scale: float) -> Tensor:
        if path.suffix.lower() == ".npy":
            depth_np = np.load(path).astype(np.float32)
            if depth_scale != self.depth_scale:
                depth_np = depth_np / depth_scale
        else:
            depth_np = np.asarray(Image.open(path), dtype=np.float32) / depth_scale
        if depth_np.ndim == 3:
            depth_np = depth_np[:, :, 0]
        return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

    def _crop_resize(
        self,
        image: Image.Image,
        depth: Tensor,
        sparse: Optional[Tensor],
        K: Tensor,
    ) -> Tuple[Image.Image, Tensor, Optional[Tensor], Tensor]:
        orig_h, orig_w = depth.shape[-2:]
        crop_h, crop_w = orig_h, orig_w

        if self.crop_box is not None:
            top, left, bottom, right = self.crop_box
            image = image.crop((left, top, right, bottom))
            depth = depth[:, top:bottom, left:right]
            if sparse is not None:
                sparse = sparse[:, top:bottom, left:right]
            K = K.clone()
            K[0, 2] -= float(left)
            K[1, 2] -= float(top)
            crop_h, crop_w = bottom - top, right - left

        if self.split == "train" and self.train_resize is not None:
            target_h, target_w = self.train_resize
        else:
            target_h, target_w = self.img_size
        if (image.height, image.width) != (target_h, target_w):
            image = image.resize((target_w, target_h), Image.BILINEAR)
        if depth.shape[-2:] != (target_h, target_w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(target_h, target_w), mode="nearest"
            ).squeeze(0)
            if sparse is not None:
                sparse = torch.nn.functional.interpolate(
                    sparse.unsqueeze(0), size=(target_h, target_w), mode="nearest"
                ).squeeze(0)

        K = K.clone()
        K[0, 0] *= float(target_w) / float(crop_w)
        K[0, 2] *= float(target_w) / float(crop_w)
        K[1, 1] *= float(target_h) / float(crop_h)
        K[1, 2] *= float(target_h) / float(crop_h)
        return image, depth, sparse, K

    def _preprocess_train(
        self,
        image: Image.Image,
        depth: Tensor,
        sparse: Optional[Tensor],
        K: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        depth_np = depth.squeeze(0).numpy()
        sparse_np = sparse.squeeze(0).numpy() if sparse is not None else None

        # PixelFormer-style random in-plane rotation. A rotation about the
        # optical axis is an extrinsic camera rotation, so the pinhole
        # intrinsics K are unchanged. Image uses bilinear; depth/sparse use
        # nearest with 0-fill, so rotated-in corners become depth 0 and are
        # dropped by valid_mask (depth > min_depth).
        if self.rotate_degree > 0 and random.random() > 0.5:
            angle = random.uniform(-self.rotate_degree, self.rotate_degree)
            image = image.rotate(angle, resample=Image.BILINEAR)
            depth_np = self._rotate_map(depth_np, angle)
            if sparse_np is not None:
                sparse_np = self._rotate_map(sparse_np, angle)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        cur_h, cur_w = image_np.shape[:2]

        if random.random() > 0.5:
            image_np = image_np[:, ::-1, :].copy()
            depth_np = depth_np[:, ::-1].copy()
            if sparse_np is not None:
                sparse_np = sparse_np[:, ::-1].copy()
            K = K.clone()
            K[0, 2] = float(cur_w) - K[0, 2]

        # zip-style random crop from the larger working size down to img_size.
        target_h, target_w = self.img_size
        if (cur_h, cur_w) != (target_h, target_w):
            top = random.randint(0, max(cur_h - target_h, 0))
            left = random.randint(0, max(cur_w - target_w, 0))
            image_np = image_np[top:top + target_h, left:left + target_w, :]
            depth_np = depth_np[top:top + target_h, left:left + target_w]
            if sparse_np is not None:
                sparse_np = sparse_np[top:top + target_h, left:left + target_w]
            K = K.clone()
            K[0, 2] -= float(left)
            K[1, 2] -= float(top)

        if random.random() > 0.5:
            image_np = self._augment_image(image_np)

        # CutDepth: paste per-image-normalised GT depth into a random full-width
        # band, only where depth is valid. Injects a strong depth cue and curbs
        # the train/val gap (AdaBins/PixelFormer use it).
        if self.cutdepth_prob > 0 and random.random() < self.cutdepth_prob:
            image_np = self._cut_depth(image_np, depth_np)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
        image_tensor = self.normalize(image_tensor)
        depth_tensor = torch.from_numpy(depth_np.copy()).float().unsqueeze(0)
        sparse_tensor = (
            torch.from_numpy(sparse_np.copy()).float().unsqueeze(0)
            if sparse_np is not None else None
        )
        return image_tensor, depth_tensor, sparse_tensor, K

    def _preprocess_eval_image(self, image: Image.Image) -> Tensor:
        if self.transform is not None:
            return self.transform(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
        return self.normalize(tensor)

    def _camera_mask(self, img_rel: str, h: int, w: int) -> Tensor:
        mask = torch.ones(1, h, w, dtype=torch.bool)
        if "CAMERA_09" not in img_rel:
            return mask
        pre_h = float(self.DEFAULT_PRE_RESIZE_SIZE[0])
        bottom = int(round(float(self.camera09_bottom_mask) / pre_h * float(h)))
        if bottom > 0:
            mask[:, h - bottom:, :] = False
        return mask

    @staticmethod
    def _augment_image(image: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        image_aug *= random.uniform(0.75, 1.25)   # outdoor brightness (PixelFormer KITTI)
        colors = np.random.uniform(0.9, 1.1, size=3)
        image_aug *= colors.reshape(1, 1, 3)
        return np.clip(image_aug, 0.0, 1.0)

    @staticmethod
    def _rotate_map(arr: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a single-channel float map by ``angle`` degrees, nearest, 0-fill."""
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        pil = pil.rotate(angle, resample=Image.NEAREST, fillcolor=0.0)
        return np.asarray(pil, dtype=np.float32)

    def _cut_depth(self, image_np: np.ndarray, depth_np: np.ndarray) -> np.ndarray:
        """Replace a random full-width band of the RGB with normalised GT depth.

        Only valid-depth pixels are overwritten; invalid (0/sky) pixels keep
        their RGB. Depth is min-max normalised per image over its valid pixels.
        Operates in place on a writable, contiguous ``image_np``.
        """
        h, w = depth_np.shape
        valid = (
            np.isfinite(depth_np)
            & (depth_np > self.min_depth)
            & (depth_np < self.max_depth)
        )
        if int(valid.sum()) < 16:
            return image_np
        dvals = depth_np[valid]
        dmin, dmax = float(dvals.min()), float(dvals.max())
        if dmax <= dmin:
            return image_np
        dnorm = np.clip((depth_np - dmin) / (dmax - dmin), 0.0, 1.0)

        band = random.randint(h // 4, h)
        top = random.randint(0, h - band)
        region_valid = valid[top:top + band, :]
        patch = image_np[top:top + band, :, :]          # view into image_np
        patch[region_valid] = dnorm[top:top + band, :][..., None][region_valid]
        return image_np
