"""Dual-teacher distillation loss."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


SHALLOW_LAYERS: List[str] = ["layer_2", "layer_5"]
DEEP_LAYERS: List[str] = ["layer_8", "layer_11"]


def dual_distill_loss(
    student_out: Dict[str, any],
    F_TA: Dict[str, Tensor],
    F_TB: Dict[str, Tensor],
    shallow_layers: List[str] = SHALLOW_LAYERS,
    deep_layers: List[str] = DEEP_LAYERS,
    lambda_boundary: float = 1.0,
    lambda_scale: float = 1.0,
) -> Tensor:
    """
    Compute dual-teacher feature distillation loss.

    L_boundary: MSE between student shallow feats and boundary teacher (F_TA).
    L_scale:    MSE between student deep feats and scale teacher (F_TB).

    Args:
        student_out: dict with keys "shallow_feats" and "deep_feats".
        F_TA: boundary teacher features keyed by layer name.
        F_TB: scale teacher features keyed by layer name.

    Returns:
        Total distillation loss scalar.
    """
    shallow_feats = student_out.get("shallow_feats", {})
    deep_feats = student_out.get("deep_feats", {})

    L_boundary = torch.tensor(0.0, device=_device_from(shallow_feats, F_TA))
    for l in shallow_layers:
        s_feat = _get_feat(shallow_feats, l)
        t_feat = F_TA.get(l)
        if s_feat is not None and t_feat is not None:
            t_feat = _align_shape(s_feat, t_feat)
            L_boundary = L_boundary + F.mse_loss(s_feat, t_feat)

    L_scale = torch.tensor(0.0, device=_device_from(deep_feats, F_TB))
    for l in deep_layers:
        s_feat = _get_feat(deep_feats, l)
        t_feat = F_TB.get(l)
        if s_feat is not None and t_feat is not None:
            t_feat = _align_shape(s_feat, t_feat)
            L_scale = L_scale + F.mse_loss(s_feat, t_feat)

    return lambda_boundary * L_boundary + lambda_scale * L_scale


def _get_feat(feat_source, key: str):
    """Get a feature by key from dict or list (index)."""
    if isinstance(feat_source, dict):
        return feat_source.get(key)
    if isinstance(feat_source, (list, tuple)):
        try:
            idx = int(key.split("_")[-1])
            return feat_source[idx] if idx < len(feat_source) else None
        except (ValueError, IndexError):
            return None
    return None


def _align_shape(src: Tensor, tgt: Tensor) -> Tensor:
    """Spatially resize tgt to match src if necessary."""
    if src.shape == tgt.shape:
        return tgt
    if src.dim() == 4 and tgt.dim() == 4:
        return F.interpolate(tgt, size=src.shape[2:], mode="bilinear", align_corners=False)
    if src.dim() == 3 and tgt.dim() == 3:
        return F.interpolate(
            tgt.unsqueeze(0), size=src.shape[1:], mode="bilinear", align_corners=False
        ).squeeze(0)
    if src.dim() == 2 and tgt.dim() == 2 and src.shape[1] != tgt.shape[1]:
        # Channel projection not done here; just truncate/pad for safety
        min_c = min(src.shape[1], tgt.shape[1])
        return tgt[:, :min_c] if tgt.shape[1] > min_c else tgt
    return tgt


def _device_from(*dicts):
    for d in dicts:
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, Tensor):
                    return v.device
        elif isinstance(d, (list, tuple)):
            for v in d:
                if isinstance(v, Tensor):
                    return v.device
    return torch.device("cpu")
