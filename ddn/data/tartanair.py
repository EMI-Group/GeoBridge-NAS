"""TartanAir dataset loader for DDN-NAS."""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from .sparse_sampler import simulate_sparse_depth


class TartanAirDataset(Dataset):
    """
    TartanAir dataset with optional 5% sparse depth simulation.

    Directory layout expected:
        root/
          <env_name>/
            <difficulty>/
              image_left/   *.png
              depth_left/   *.npy

    Args:
        root: path to TartanAir root directory.
        split: "train" | "val" | "test".
        sparse_ratio: if > 0, simulate sparse depth at this fraction.
        transform: image transform (default: normalize to ImageNet stats).
        max_samples: cap dataset size (useful for 10% subset).
        cache_dir: if set, path to pre-computed feature cache directory.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "train",
        sparse_ratio: float = 0.05,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        img_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.sparse_ratio = sparse_ratio
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.img_size = img_size  # (H, W)

        self.transform = transform or T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        self.samples: List[Tuple[Path, Path]] = self._collect_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        samples = []
        split_file = self.root / f"{self.split}_split.txt"

        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    img_path, depth_path = line.strip().split()
                    samples.append((self.root / img_path, self.root / depth_path))
        else:
            # Auto-discover: walk directory tree
            # TartanAir layout: <env>/<difficulty>/<trajectory>/image_left/
            for env_dir in sorted(self.root.iterdir()):
                if not env_dir.is_dir():
                    continue
                for diff_dir in sorted(env_dir.iterdir()):
                    if not diff_dir.is_dir():
                        continue
                    for traj_dir in sorted(diff_dir.iterdir()):
                        if not traj_dir.is_dir():
                            continue
                        img_dir = traj_dir / "image_left"
                        dep_dir = traj_dir / "depth_left"
                        if not img_dir.exists() or not dep_dir.exists():
                            continue
                        img_files = sorted(img_dir.glob("*.png"))
                        for img_file in img_files:
                            # TartanAir depth files: 000000_left.png → 000000_left_depth.npy
                            stem = img_file.stem  # e.g. "000000_left"
                            dep_file = dep_dir / f"{stem}_depth.npy"
                            if dep_file.exists():
                                samples.append((img_file, dep_file))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, dep_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        depth = np.load(dep_path).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        depth = self._resize_depth(depth)

        # TartanAir sky/invalid pixels are very large floats; clamp to valid range
        depth = torch.clamp(depth, min=0.0, max=1000.0)

        result = {
            "image": img_tensor,       # (3, H, W)
            "depth": depth,            # (1, H, W) dense GT
            "idx": idx,
            "img_path": str(img_path),
        }

        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{idx:08d}.pt"
            if cache_file.exists():
                result["cached_features"] = torch.load(cache_file, weights_only=True)

        if self.sparse_ratio > 0:
            result["sparse_depth"] = simulate_sparse_depth(depth, self.sparse_ratio)

        return result

    def _resize_depth(self, depth: Tensor) -> Tensor:
        H, W = self.img_size
        if depth.shape[-2:] != (H, W):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)
        return depth
