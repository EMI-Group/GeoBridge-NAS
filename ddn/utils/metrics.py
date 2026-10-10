"""Depth estimation and boundary evaluation metrics."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_depth_metrics(
    pred: Tensor,
    gt: Tensor,
    mask: Optional[Tensor] = None,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.

    Args:
        pred: (B, 1, H, W) or (B, H, W) predicted depth.
        gt:   (B, 1, H, W) or (B, H, W) ground truth depth.
        mask: optional valid-pixel mask (same shape, bool).

    Returns:
        Dict with AbsRel, SqRel, RMSE, RMSElog, δ₁, δ₂, δ₃.
    """
    pred = pred.squeeze(1).float()
    gt = gt.squeeze(1).float()

    if mask is None:
        mask = (gt > min_depth) & (gt < max_depth)
    else:
        mask = mask.squeeze(1).bool()

    pred = pred[mask].clamp(min_depth, max_depth)
    gt = gt[mask].clamp(min_depth, max_depth)

    if pred.numel() == 0:
        return {k: float("nan") for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]}

    thresh = torch.max(pred / gt, gt / pred)
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_rel = ((pred - gt).abs() / gt).mean().item()
    sq_rel = (((pred - gt) ** 2) / gt).mean().item()
    rmse = ((pred - gt) ** 2).mean().sqrt().item()
    rmse_log = ((pred.log() - gt.log()) ** 2).mean().sqrt().item()

    return {
        "abs_rel": abs_rel,
        "sq_rel":  sq_rel,
        "rmse":    rmse,
        "rmse_log": rmse_log,
        "d1": d1,
        "d2": d2,
        "d3": d3,
    }


def compute_boundary_fscore(
    pred_depth: Tensor,
    gt_depth: Tensor,
    threshold_px: int = 1,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Boundary F-score: measures sharpness of depth discontinuities.

    Extracts edges from pred and gt via Sobel, then computes precision/recall
    at `threshold_px` pixel tolerance.

    Args:
        pred_depth: (B, 1, H, W) predicted depth.
        gt_depth:   (B, 1, H, W) ground truth depth.
        threshold_px: spatial tolerance for edge matching.

    Returns:
        F-score scalar in [0, 1].
    """
    pred_edges = _sobel_edges(pred_depth)  # (B, 1, H, W)
    gt_edges = _sobel_edges(gt_depth)

    if mask is not None:
        pred_edges = pred_edges * mask
        gt_edges = gt_edges * mask

    # Dilate gt edges by threshold_px for tolerance matching
    gt_dilated = _dilate(gt_edges, radius=threshold_px)
    pred_dilated = _dilate(pred_edges, radius=threshold_px)

    tp_p = (pred_edges * gt_dilated).sum()
    tp_r = (gt_edges * pred_dilated).sum()

    precision = tp_p / (pred_edges.sum() + 1e-6)
    recall = tp_r / (gt_edges.sum() + 1e-6)

    if precision + recall < 1e-6:
        return 0.0
    f = (2 * precision * recall / (precision + recall)).item()
    return f


def _sobel_edges(depth: Tensor) -> Tensor:
    """Compute binary edge map using Sobel filter."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth.dtype, device=depth.device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    gx = F.conv2d(depth, sobel_x, padding=1)
    gy = F.conv2d(depth, sobel_y, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt()

    # Threshold at mean + std
    threshold = mag.mean() + mag.std()
    return (mag > threshold).float()


def _dilate(binary: Tensor, radius: int) -> Tensor:
    """Binary dilation with square structuring element."""
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=binary.device)
    dilated = F.conv2d(binary, kernel, padding=radius)
    return (dilated > 0).float()
