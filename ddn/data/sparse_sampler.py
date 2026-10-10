"""Sparse depth simulation for NLSPN teacher training."""

import torch
import numpy as np
from torch import Tensor


def simulate_sparse_depth(gt_depth: Tensor, ratio: float = 0.05) -> Tensor:
    """
    Simulate LiDAR sparse depth from dense GT depth.

    Uses Poisson disk sampling to mimic LiDAR scan-line patterns.

    Args:
        gt_depth: (B, 1, H, W) or (1, H, W) or (H, W) dense depth map.
        ratio: fraction of pixels to keep (default 5%).

    Returns:
        Sparse depth tensor of the same shape; zero where no measurement.
    """
    squeeze_dims = []
    if gt_depth.dim() == 2:
        gt_depth = gt_depth.unsqueeze(0).unsqueeze(0)
        squeeze_dims = [0, 1]
    elif gt_depth.dim() == 3:
        gt_depth = gt_depth.unsqueeze(0)
        squeeze_dims = [0]

    B, C, H, W = gt_depth.shape
    sparse = torch.zeros_like(gt_depth)

    for b in range(B):
        mask = _poisson_disk_mask(H, W, ratio, device=gt_depth.device)
        sparse[b] = gt_depth[b] * mask

    for dim in reversed(squeeze_dims):
        sparse = sparse.squeeze(dim)
    return sparse


def _poisson_disk_mask(H: int, W: int, ratio: float, device) -> Tensor:
    """
    Generate a binary mask approximating Poisson disk sampling.

    Approximation: stratified random sampling within a grid, which is faster
    than true Poisson disk sampling and sufficient for simulating LiDAR density.
    """
    n_points = max(1, int(H * W * ratio))

    # Stratified sampling across horizontal scan lines (LiDAR-like)
    n_lines = max(1, int(np.sqrt(n_points)))
    line_indices = np.linspace(0, H - 1, n_lines, dtype=int)

    mask = np.zeros((1, H, W), dtype=np.float32)
    pts_per_line = max(1, n_points // n_lines)

    for row in line_indices:
        cols = np.random.choice(W, size=min(pts_per_line, W), replace=False)
        mask[0, row, cols] = 1.0

    return torch.from_numpy(mask).to(device)


def random_sparse_depth(gt_depth: Tensor, ratio: float = 0.05) -> Tensor:
    """Uniform random sparse sampling (simpler alternative to Poisson disk)."""
    mask = torch.zeros_like(gt_depth)
    n = max(1, int(gt_depth.numel() * ratio))
    flat_idx = torch.randperm(gt_depth.numel(), device=gt_depth.device)[:n]
    mask.view(-1)[flat_idx] = 1.0
    return gt_depth * mask
