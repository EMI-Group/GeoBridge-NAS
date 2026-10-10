"""Matterport3D RGB-D dataset loader for downstream evaluation."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .sparse_sampler import simulate_sparse_depth


class Matterport3DDataset(Dataset):
    """
    Matterport3D split-file loader.

    Each split entry is expected to be one of:
        <rgb> <depth>
        <rgb> <depth> <sparse_depth>
        <rgb> <depth> <depth_scale>
        <rgb> <depth> <fx> <fy> <cx> <cy>
        <rgb> <depth> <fx> <fy> <cx> <cy> <depth_scale>
        <rgb> <depth> <k00> <k01> <k02> <k10> <k11> <k12> <k20> <k21> <k22>
        <rgb> <depth> <k00> ... <k22> <depth_scale>

    Relative paths are resolved under ``root``.  Depth PNGs are divided by
    ``depth_scale`` to convert to metres; .npy depths are assumed to already be
    metric unless a per-row scale is supplied.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        filenames_file: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        img_size: Tuple[int, int] = (480, 640),
        depth_scale: float = 4000.0,
        max_depth: float = 10.0,
        sparse_ratio: float = 0.0,
        return_color: bool = False,
        require_intrinsics: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.filenames_file = Path(filenames_file)
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.sparse_ratio = sparse_ratio
        self.return_color = return_color
        self.require_intrinsics = require_intrinsics
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        self.transform = transform

        if not self.filenames_file.exists():
            raise FileNotFoundError(self.filenames_file)

        with open(self.filenames_file) as f:
            self.samples: List[List[str]] = [
                line.strip().split()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fields = self.samples[idx]
        valid_with_intrinsics = len(fields) in {6, 7, 11, 12}
        valid_without_intrinsics = not self.require_intrinsics and len(fields) in {2, 3}
        if not valid_with_intrinsics and not valid_without_intrinsics:
            raise ValueError(
                "Matterport3D split entries must be rgb/depth plus fx fy cx cy "
                "or a full 3x3 K matrix, with optional depth_scale. "
                f"Got {len(fields)} columns at index {idx}: {' '.join(fields[:8])}"
            )

        img_path = self._resolve(fields[0])
        depth_path = self._resolve(fields[1])
        K = self._parse_intrinsics(fields) if valid_with_intrinsics else torch.eye(3)
        sample_scale = self._parse_depth_scale(fields)
        sparse_path = self._parse_sparse_depth_path(fields)

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        depth = self._load_depth(depth_path, sample_scale)
        sparse_depth = self._load_depth(sparse_path, sample_scale) if sparse_path is not None else None

        image_tensor, color_tensor = self._preprocess_image(image)
        depth = self._resize_depth(depth)
        if sparse_depth is not None:
            sparse_depth = self._resize_depth(sparse_depth)
        K = self._scale_intrinsics(K, orig_h, orig_w)

        valid_mask = torch.isfinite(depth) & (depth > 0.0) & (depth < self.max_depth)
        depth = torch.clamp(depth, min=0.0, max=self.max_depth)
        result = {
            "image": image_tensor,
            "depth": depth,
            "valid_mask": valid_mask,
            "K": K,
            "idx": idx,
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }
        if sparse_depth is not None:
            result["sparse_depth"] = torch.clamp(sparse_depth, min=0.0, max=self.max_depth)
        elif self.sparse_ratio > 0:
            result["sparse_depth"] = simulate_sparse_depth(depth, self.sparse_ratio)
        if self.return_color:
            result["color"] = color_tensor
        return result

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def _parse_intrinsics(self, fields: List[str]) -> Tensor:
        nums = [float(x) for x in fields[2:]]
        if len(fields) in {6, 7}:
            fx, fy, cx, cy = nums[:4]
            return torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        return torch.tensor(nums[:9], dtype=torch.float32).view(3, 3)

    def _parse_depth_scale(self, fields: List[str]) -> float:
        if len(fields) == 3 and not self.require_intrinsics and self._looks_like_number(fields[2]):
            return float(fields[2])
        if len(fields) == 7:
            return float(fields[6])
        if len(fields) == 12:
            return float(fields[11])
        return self.depth_scale

    def _parse_sparse_depth_path(self, fields: List[str]) -> Optional[Path]:
        if len(fields) == 3 and not self.require_intrinsics and not self._looks_like_number(fields[2]):
            return self._resolve(fields[2])
        return None

    @staticmethod
    def _looks_like_number(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

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

    def _preprocess_image(self, image: Image.Image) -> Tuple[Tensor, Tensor]:
        h, w = self.img_size
        image = image.resize((w, h), Image.BILINEAR)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
            image_tensor = self.normalize(image_tensor)

        color_np = np.asarray(image, dtype=np.uint8)
        color_tensor = torch.from_numpy(color_np.transpose(2, 0, 1).copy())
        return image_tensor, color_tensor

    def _resize_depth(self, depth: Tensor) -> Tensor:
        h, w = self.img_size
        if depth.shape[-2:] != (h, w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)
        return depth

    def _scale_intrinsics(self, K: Tensor, orig_h: int, orig_w: int) -> Tensor:
        target_h, target_w = self.img_size
        sx = float(target_w) / float(orig_w)
        sy = float(target_h) / float(orig_h)
        K = K.clone().float()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K
