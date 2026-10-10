"""NYU Depth V2 dataset loader for retraining."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import random
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 loader compatible with evonas-style split files.

    Train split entries may be either:
        <rgb> <depth>
        <rgb> <raw_depth> <dense_depth>

    Eval split entries may be:
        <rgb> <depth>
        <rgb> <depth> <focal>

    Depth PNG files are assumed to be millimetres and are converted to metres.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        filenames_file: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        img_size: Tuple[int, int] = (480, 640),
        depth_scale: float = 1000.0,
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.root = Path(root)
        self.filenames_file = Path(filenames_file)
        self.split = split
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

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
        if len(fields) < 2:
            raise ValueError(f"Invalid NYU split entry: {' '.join(fields)}")

        img_path = self._resolve(fields[0])
        depth_path = self._target_depth_path(fields)

        img = Image.open(img_path).convert("RGB")
        depth = self._load_depth(depth_path)

        sparse_path = self._sparse_depth_path(fields)
        sparse = None
        if sparse_path is not None:
            sparse = self._load_depth(sparse_path)

        if self.split == "train":
            img_tensor, depth, sparse = self._preprocess_train(img, depth, sparse)
        else:
            img_tensor = self.transform(img)
            depth = self._resize_depth(depth)
            if sparse is not None:
                sparse = self._resize_depth(sparse)

        depth = torch.clamp(depth, min=0.0, max=self.max_depth)
        result = {
            "image": img_tensor,
            "depth": depth,
            "idx": idx,
            "img_path": str(img_path),
        }
        if sparse is not None:
            sparse = torch.clamp(sparse, min=0.0, max=self.max_depth)
            result["sparse_depth"] = sparse

        return result

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        prefix = "sync" if self.split == "train" else "test"
        return self.root / prefix / p

    def _target_depth_path(self, fields: List[str]) -> Path:
        if self.split == "train" and len(fields) >= 3 and not self._looks_like_number(fields[2]):
            return self._resolve(fields[2])
        return self._resolve(fields[1])

    def _sparse_depth_path(self, fields: List[str]) -> Optional[Path]:
        if self.split == "train" and len(fields) >= 3 and not self._looks_like_number(fields[2]):
            return self._resolve(fields[1])
        return None

    @staticmethod
    def _looks_like_number(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _load_depth(self, path: Path) -> Tensor:
        ext = path.suffix.lower()
        if ext == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.asarray(Image.open(path), dtype=np.float32) / self.depth_scale
            if depth.ndim == 3:
                depth = depth[:, :, 0]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)

    def _resize_depth(self, depth: Tensor) -> Tensor:
        h, w = self.img_size
        if depth.shape[-2:] != (h, w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)
        return depth

    def _preprocess_train(
        self,
        image: Image.Image,
        depth: Tensor,
        sparse: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        depth_np = depth.squeeze(0).numpy()
        sparse_np = sparse.squeeze(0).numpy() if sparse is not None else None

        target_h, target_w = self.img_size
        if image_np.shape[0] != target_h or image_np.shape[1] != target_w:
            image_np, depth_np, sparse_np = self._random_crop_or_resize(
                image_np, depth_np, sparse_np, target_h, target_w
            )

        if random.random() > 0.5:
            image_np = image_np[:, ::-1, :].copy()
            depth_np = depth_np[:, ::-1].copy()
            if sparse_np is not None:
                sparse_np = sparse_np[:, ::-1].copy()

        if random.random() > 0.5:
            image_np = self._augment_image(image_np)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
        image_tensor = self.normalize(image_tensor)
        depth_tensor = torch.from_numpy(depth_np.copy()).float().unsqueeze(0)
        sparse_tensor = (
            torch.from_numpy(sparse_np.copy()).float().unsqueeze(0)
            if sparse_np is not None else None
        )
        return image_tensor, depth_tensor, sparse_tensor

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
            depth = depth[y:y + target_h, x:x + target_w]
            if sparse is not None:
                sparse = sparse[y:y + target_h, x:x + target_w]
            return image, depth, sparse

        image_pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
        image = np.asarray(
            image_pil.resize((target_w, target_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        depth = torch.nn.functional.interpolate(
            torch.from_numpy(depth).view(1, 1, h, w),
            size=(target_h, target_w),
            mode="nearest",
        ).view(target_h, target_w).numpy()
        if sparse is not None:
            sparse = torch.nn.functional.interpolate(
                torch.from_numpy(sparse).view(1, 1, h, w),
                size=(target_h, target_w),
                mode="nearest",
            ).view(target_h, target_w).numpy()
        return image, depth, sparse

    @staticmethod
    def _augment_image(image: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        image_aug *= random.uniform(0.75, 1.25)
        colors = np.random.uniform(0.9, 1.1, size=3)
        image_aug *= colors.reshape(1, 1, 3)
        return np.clip(image_aug, 0.0, 1.0)
