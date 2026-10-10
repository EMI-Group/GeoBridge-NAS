"""Geometry utilities for depth-map backprojection and point-cloud metrics."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


def backproject_depth_to_points(
    depth: Tensor,
    K: Tensor,
    mask: Optional[Tensor] = None,
    stride: int = 1,
) -> Tensor:
    """
    Backproject a single depth map to camera-frame XYZ points.

    Args:
        depth: (1, H, W) or (H, W) metric depth.
        K: (3, 3) camera intrinsic matrix for the depth resolution.
        mask: optional valid-pixel mask, broadcastable to (H, W).
        stride: subsample image pixels before projection.
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    if depth.dim() != 2:
        raise ValueError(f"depth must be (H, W) or (1, H, W), got {tuple(depth.shape)}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3, 3), got {tuple(K.shape)}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    depth = depth.float()
    K = K.to(device=depth.device, dtype=depth.dtype)
    if stride > 1:
        depth = depth[::stride, ::stride]
        if mask is not None:
            mask = mask.squeeze()[::stride, ::stride]

    h, w = depth.shape
    v, u = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    if stride > 1:
        u = u * stride
        v = v * stride

    valid = torch.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid = valid & mask.squeeze().to(device=depth.device, dtype=torch.bool)

    z = depth[valid]
    if z.numel() == 0:
        return depth.new_zeros((0, 3))

    fx = K[0, 0].clamp_min(1e-8)
    fy = K[1, 1].clamp_min(1e-8)
    cx = K[0, 2]
    cy = K[1, 2]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return torch.stack([x, y, z], dim=-1)


def subsample_points(points: Tensor, max_points: Optional[int]) -> Tensor:
    """Deterministically subsample a point cloud to at most ``max_points``."""
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=max_points,
        device=points.device,
    ).long()
    return points[idx]


def compute_pointcloud_metrics(
    pred_points: Tensor,
    gt_points: Tensor,
    thresholds: Iterable[float] = (0.05, 0.10),
    max_points: Optional[int] = 20000,
    chunk_size: int = 4096,
) -> Dict[str, float]:
    """
    Compute standard point-cloud reconstruction metrics.

    Returns Chamfer-L1/L2, one-way accuracy/completeness, and precision,
    recall, F-score for each distance threshold.
    """
    pred_points = subsample_points(pred_points.float(), max_points)
    gt_points = subsample_points(gt_points.float(), max_points)
    keys = ["chamfer_l1", "chamfer_l2", "accuracy", "completeness"]

    if pred_points.numel() == 0 or gt_points.numel() == 0:
        out = {k: float("nan") for k in keys}
        for t in thresholds:
            suffix = _threshold_suffix(t)
            out[f"precision_{suffix}"] = float("nan")
            out[f"recall_{suffix}"] = float("nan")
            out[f"fscore_{suffix}"] = float("nan")
        return out

    d_pred_gt = _nearest_distances(pred_points, gt_points, chunk_size=chunk_size)
    d_gt_pred = _nearest_distances(gt_points, pred_points, chunk_size=chunk_size)

    out = {
        "chamfer_l1": 0.5 * (d_pred_gt.mean() + d_gt_pred.mean()).item(),
        "chamfer_l2": 0.5 * (d_pred_gt.pow(2).mean() + d_gt_pred.pow(2).mean()).item(),
        "accuracy": d_pred_gt.mean().item(),
        "completeness": d_gt_pred.mean().item(),
    }
    for t in thresholds:
        precision = (d_pred_gt < t).float().mean()
        recall = (d_gt_pred < t).float().mean()
        fscore = 2.0 * precision * recall / (precision + recall + 1e-8)
        suffix = _threshold_suffix(t)
        out[f"precision_{suffix}"] = precision.item()
        out[f"recall_{suffix}"] = recall.item()
        out[f"fscore_{suffix}"] = fscore.item()
    return out


def save_point_cloud_ply(
    path: str,
    points: Tensor,
    colors: Optional[Tensor] = None,
) -> None:
    """Write an ASCII PLY point cloud with optional uint8 RGB colors."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    points_cpu = points.detach().cpu().float()
    colors_cpu = None
    if colors is not None:
        colors_cpu = colors.detach().cpu().clamp(0, 255).byte()
        if colors_cpu.shape[0] != points_cpu.shape[0]:
            raise ValueError("colors must have the same number of rows as points")

    with open(path_obj, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points_cpu.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors_cpu is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(points_cpu):
            if colors_cpu is None:
                f.write(f"{p[0].item():.6f} {p[1].item():.6f} {p[2].item():.6f}\n")
            else:
                c = colors_cpu[i]
                f.write(
                    f"{p[0].item():.6f} {p[1].item():.6f} {p[2].item():.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )


def _nearest_distances(src: Tensor, tgt: Tensor, chunk_size: int) -> Tensor:
    mins = []
    for start in range(0, src.shape[0], chunk_size):
        chunk = src[start : start + chunk_size]
        mins.append(torch.cdist(chunk, tgt).min(dim=1).values)
    return torch.cat(mins, dim=0)


def _threshold_suffix(threshold: float) -> str:
    return f"{int(round(threshold * 1000)):03d}mm"
