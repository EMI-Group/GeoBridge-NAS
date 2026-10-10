"""Grad-Sensitivity proxy metric (quantity × quality)."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distill_loss import dual_distill_loss, SHALLOW_LAYERS, DEEP_LAYERS


def compute_grad_sensitivity(
    student: nn.Module,
    batch: Tensor,
    F_TA: Dict[str, Tensor],
    F_TB: Dict[str, Tensor],
    shallow_layers: List[str] = SHALLOW_LAYERS,
    deep_layers: List[str] = DEEP_LAYERS,
    decoder_attr: str = "decoder",
    depth_target: Optional[Tensor] = None,
) -> float:
    """
    Compute gradient sensitivity score for a student decoder.

    Score = sensitivity × |alignment|
    - sensitivity: average |grad * param| across decoder parameters.
    - alignment:   cosine similarity between flattened grad and param vectors.

    Args:
        student: StudentNet with a `decoder` attribute.
        batch: (B, 3, H, W) image batch (on the same device as student).
        F_TA: boundary teacher features dict.
        F_TB: scale teacher features dict.
        decoder_attr: name of the decoder attribute on student.

    Returns:
        Non-negative float grad-sensitivity score.
    """
    # Switch to train mode: DINOv2Encoder wraps forward in no_grad when eval.
    was_training = student.training
    student.train()
    student.zero_grad()

    # Use forward_decoder_only when available: runs encoder under no_grad so
    # only scale_proj + decoder graph is kept in memory (~100× less VRAM).
    if hasattr(student, "forward_decoder_only"):
        out = student.forward_decoder_only(batch)
    else:
        out = student(batch)

    depth = out.get("depth")
    if depth is not None and depth.requires_grad:
        if depth_target is not None:
            target = depth_target.to(device=depth.device, dtype=depth.dtype)
            if target.shape[-2:] != depth.shape[-2:]:
                target = F.interpolate(target, size=depth.shape[-2:], mode="bilinear", align_corners=False)
            target = target.expand_as(depth).detach()
            L = torch.nn.functional.mse_loss(depth, target)
        else:
            t_depth = F_TA.get(shallow_layers[0]) if F_TA else None
            if t_depth is not None and t_depth.dim() == 2:
                target = (
                    t_depth.mean(dim=1, keepdim=True)
                    .unsqueeze(-1).unsqueeze(-1)
                    .expand_as(depth).detach()
                )
                L = torch.nn.functional.mse_loss(depth, target)
            else:
                L = depth.mean()
    else:
        L = dual_distill_loss(out, F_TA, F_TB, shallow_layers, deep_layers)

    if not L.requires_grad:
        if not was_training:
            student.eval()
        return 0.0

    L.backward()

    if not was_training:
        student.eval()

    decoder = getattr(student, decoder_attr)
    grads: List[Tensor] = []
    params: List[Tensor] = []

    for p in decoder.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach())
            params.append(p.detach())

    if not grads:
        return 0.0

    # Quantity: mean |g * p|
    total_sensitivity = sum(
        (g.abs() * p.abs()).sum() for g, p in zip(grads, params)
    )
    n_params = sum(p.numel() for p in params)
    sensitivity = (total_sensitivity / n_params).item()

    # Quality: cosine alignment between flattened grad and param
    g_flat = torch.cat([g.flatten() for g in grads])
    p_flat = torch.cat([p.flatten() for p in params])
    alignment = F.cosine_similarity(
        g_flat.unsqueeze(0), p_flat.unsqueeze(0)
    ).item()

    return sensitivity * abs(alignment)


class GradSensitivityScorer:
    """Stateful scorer that accumulates sensitivity over multiple batches."""

    def __init__(
        self,
        shallow_layers: List[str] = SHALLOW_LAYERS,
        deep_layers: List[str] = DEEP_LAYERS,
    ):
        self.shallow_layers = shallow_layers
        self.deep_layers = deep_layers
        self._scores: List[float] = []

    def score(
        self,
        student: nn.Module,
        batch: Tensor,
        F_TA: Dict[str, Tensor],
        F_TB: Dict[str, Tensor],
    ) -> float:
        s = compute_grad_sensitivity(
            student, batch, F_TA, F_TB, self.shallow_layers, self.deep_layers
        )
        self._scores.append(s)
        return s

    def mean(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def reset(self) -> None:
        self._scores.clear()
