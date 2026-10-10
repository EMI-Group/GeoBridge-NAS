"""KITTI Eigen split dataset loader with evonas-style augmentation."""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .sparse_sampler import simulate_sparse_depth


class KITTIEigenDataset(Dataset):
    """
    KITTI Eigen split loader.

    Split entries follow the evonas convention:
        <rgb> <dense_depth> [focal]
        <rgb> <dense_depth> <sparse_depth> [focal]

    Relative RGB paths are resolved under ``root``. Relative depth/sparse paths
    are resolved under ``gt_root`` when provided, otherwise under ``root``.
    Depth PNG values are divided by ``depth_scale`` (KITTI default: 256).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEFAULT_FOCAL = 721.5377

    def __init__(
        self,
        root: str,
        filenames_file: str,
        split: str = "train",
        gt_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        img_size: Tuple[int, int] = (352, 1216),
        depth_scale: float = 256.0,
        min_depth: float = 0.0,
        max_depth: float = 80.0,
        sparse_ratio: float = 0.0,
        do_kb_crop: bool = True,
        do_random_rotate: bool = False,
        degree: float = 2.5,
        use_right: bool = False,
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
        self.do_kb_crop = do_kb_crop
        self.do_random_rotate = do_random_rotate
        self.degree = degree
        self.use_right = use_right
        self.transform = transform
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

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
        self.samples = samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fields = list(self.samples[idx])
        if len(fields) < 2:
            raise ValueError(f"Invalid KITTI split entry: {' '.join(fields)}")

        if self.split == "train" and self.use_right and random.random() > 0.5:
            fields[0] = fields[0].replace("image_02", "image_03")
            fields[1] = fields[1].replace("image_02", "image_03")
            if len(fields) >= 3 and not self._looks_like_number(fields[2]):
                fields[2] = fields[2].replace("image_02", "image_03")

        img_path = self._resolve(self.root, fields[0])
        depth_path = self._resolve_kitti_path(self.gt_root, fields[1])
        image = Image.open(img_path).convert("RGB")
        depth_np = self._load_depth(depth_path)

        sparse_np = None
        sparse_field = self._sparse_depth_field(fields)
        if sparse_field is not None:
            sparse_np = self._load_depth(self._resolve_kitti_path(self.gt_root, sparse_field))

        focal = self._parse_focal(fields)

        if self.do_kb_crop:
            image, depth_np, sparse_np = self._kb_crop(image, depth_np, sparse_np)

        if self.split == "train":
            image_tensor, depth, sparse = self._preprocess_train(image, depth_np, sparse_np)
        else:
            image_tensor = self._preprocess_eval_image(image)
            depth = self._resize_depth(torch.from_numpy(depth_np[:, :, 0]).unsqueeze(0))
            sparse = (
                self._resize_depth(torch.from_numpy(sparse_np[:, :, 0]).unsqueeze(0))
                if sparse_np is not None else None
            )

        depth = depth.float()
        valid_mask = (
            torch.isfinite(depth)
            & (depth > self.min_depth)
            & (depth < self.max_depth)
        )
        depth = torch.clamp(depth, min=0.0, max=self.max_depth)
        result = {
            "image": image_tensor,
            "depth": depth,
            "valid_mask": valid_mask,
            "focal": torch.tensor(float(focal), dtype=torch.float32),
            "idx": idx,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }

        if sparse is not None:
            result["sparse_depth"] = torch.clamp(sparse.float(), min=0.0, max=self.max_depth)
        elif self.sparse_ratio > 0:
            result["sparse_depth"] = simulate_sparse_depth(depth, self.sparse_ratio)
        return result

    def _resolve(self, root: Path, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else root / p

    def _resolve_kitti_path(self, root: Path, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if not p.parts:
            return root / p

        first = p.parts[0]
        date = self._date_from_drive_dir(first)
        if date is None:
            return root / p

        # Split files reference GT as <drive>/proj_depth/... with no prefix,
        # but on-disk layouts differ: BTS-style trees insert a date level,
        # the official data_depth_annotated download buckets drives under
        # train/ and val/. Probe the known layouts and take the first hit;
        # fall back to the date-inserted path so a genuinely missing file
        # still fails loudly with a sensible candidate in the message.
        candidates = [root / p]
        if root.name != date:
            candidates.append(root / date / p)
        candidates.extend([root / "train" / p, root / "val" / p])
        for cand in candidates:
            if cand.exists():
                return cand
        return root / date / p if root.name != date else root / p

    @staticmethod
    def _date_from_drive_dir(dirname: str) -> Optional[str]:
        if len(dirname) < 10:
            return None
        date = dirname[:10]
        if (
            len(date) == 10
            and date[4] == "_"
            and date[7] == "_"
            and date.replace("_", "").isdigit()
            and dirname.startswith(f"{date}_drive_")
        ):
            return date
        return None

    def _load_depth(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.asarray(Image.open(path), dtype=np.float32) / self.depth_scale
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return np.expand_dims(depth.astype(np.float32), axis=2)

    def _sparse_depth_field(self, fields: List[str]) -> Optional[str]:
        if len(fields) >= 3 and not self._looks_like_number(fields[2]):
            return fields[2]
        return None

    def _parse_focal(self, fields: List[str]) -> float:
        for value in reversed(fields[2:]):
            if self._looks_like_number(value):
                return float(value)
        return self.DEFAULT_FOCAL

    @staticmethod
    def _looks_like_number(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _kb_crop(
        self,
        image: Image.Image,
        depth: np.ndarray,
        sparse: Optional[np.ndarray],
    ) -> Tuple[Image.Image, np.ndarray, Optional[np.ndarray]]:
        # Exact BTS/evonas semantics: anchor 352x1216 at bottom-center of the
        # image. We don't clamp `right`/`bottom` so a too-small image fails the
        # depth slice loudly instead of silently producing odd shapes.
        top = max(int(image.height - 352), 0)
        left = max(int((image.width - 1216) / 2), 0)
        image = image.crop((left, top, left + 1216, top + 352))
        depth = depth[top:top + 352, left:left + 1216, :]
        if sparse is not None:
            sparse = sparse[top:top + 352, left:left + 1216, :]
        return image, depth, sparse

    def _preprocess_train(
        self,
        image: Image.Image,
        depth: np.ndarray,
        sparse: Optional[np.ndarray],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if self.do_random_rotate:
            angle = (random.random() - 0.5) * 2.0 * self.degree
            image = image.rotate(angle, resample=Image.BILINEAR)
            depth = self._rotate_depth_array(depth, angle)
            if sparse is not None:
                sparse = self._rotate_depth_array(sparse, angle)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        target_h, target_w = self.img_size
        if image_np.shape[0] != target_h or image_np.shape[1] != target_w:
            image_np, depth, sparse = self._random_crop_or_resize(
                image_np, depth, sparse, target_h, target_w
            )

        if random.random() > 0.5:
            image_np = image_np[:, ::-1, :].copy()
            depth = depth[:, ::-1, :].copy()
            if sparse is not None:
                sparse = sparse[:, ::-1, :].copy()

        if random.random() > 0.5:
            image_np = self._augment_image(image_np)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
        image_tensor = self.normalize(image_tensor)
        depth_tensor = torch.from_numpy(depth[:, :, 0].copy()).float().unsqueeze(0)
        sparse_tensor = (
            torch.from_numpy(sparse[:, :, 0].copy()).float().unsqueeze(0)
            if sparse is not None else None
        )
        return image_tensor, depth_tensor, sparse_tensor

    def _preprocess_eval_image(self, image: Image.Image) -> Tensor:
        # Skip the no-op PIL resize when kb_crop has already produced the eval
        # size; only resize when the image actually differs from img_size.
        h, w = self.img_size
        if (image.height, image.width) != (h, w):
            image = image.resize((w, h), Image.BILINEAR)
        if self.transform is not None:
            return self.transform(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
        return self.normalize(tensor)

    def _resize_depth(self, depth: Tensor) -> Tensor:
        h, w = self.img_size
        if depth.shape[-2:] != (h, w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)
        return depth

    @staticmethod
    def _random_crop_or_resize(
        image: np.ndarray,
        depth: np.ndarray,
        sparse: Optional[np.ndarray],
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w = image.shape[:2]
        if h >= target_h and w >= target_w:
            x = random.randint(0, w - target_w)
            y = random.randint(0, h - target_h)
            image = image[y:y + target_h, x:x + target_w, :]
            depth = depth[y:y + target_h, x:x + target_w, :]
            if sparse is not None:
                sparse = sparse[y:y + target_h, x:x + target_w, :]
            return image, depth, sparse

        image_pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
        image = np.asarray(image_pil.resize((target_w, target_h), Image.BILINEAR), dtype=np.float32) / 255.0
        depth = torch.nn.functional.interpolate(
            torch.from_numpy(depth[:, :, 0]).view(1, 1, h, w),
            size=(target_h, target_w),
            mode="nearest",
        ).view(target_h, target_w).numpy()[:, :, None]
        if sparse is not None:
            sparse = torch.nn.functional.interpolate(
                torch.from_numpy(sparse[:, :, 0]).view(1, 1, h, w),
                size=(target_h, target_w),
                mode="nearest",
            ).view(target_h, target_w).numpy()[:, :, None]
        return image, depth, sparse

    @staticmethod
    def _rotate_depth_array(depth: np.ndarray, angle: float) -> np.ndarray:
        depth_img = Image.fromarray(depth[:, :, 0].astype(np.float32), mode="F")
        depth_img = depth_img.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return np.expand_dims(np.asarray(depth_img, dtype=np.float32), axis=2)

    @staticmethod
    def _augment_image(image: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        image_aug *= random.uniform(0.9, 1.1)
        colors = np.random.uniform(0.9, 1.1, size=3)
        image_aug *= colors.reshape(1, 1, 3)
        return np.clip(image_aug, 0.0, 1.0)
