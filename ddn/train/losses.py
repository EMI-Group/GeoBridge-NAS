"""Training losses for full retraining phase."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..proxy.distill_loss import dual_distill_loss, SHALLOW_LAYERS, DEEP_LAYERS


class NumericalHealthError(RuntimeError):
    """Raised when a loss input cannot produce a numerically valid update."""

    def __init__(self, check: str, details: Dict[str, Any]):
        self.check = check
        self.details = details
        fields = ", ".join(f"{key}={value}" for key, value in details.items())
        super().__init__(f"{check}: {fields}")


def _as_4d_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    try:
        return torch.broadcast_to(mask.bool(), reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}"
        ) from exc


def _require_finite_prediction(pred: Tensor) -> None:
    finite = torch.isfinite(pred)
    if finite.all():
        return
    nonfinite = int((~finite).sum().item())
    raise NumericalHealthError(
        "nonfinite_prediction",
        {
            "shape": tuple(pred.shape),
            "dtype": str(pred.dtype),
            "nonfinite": nonfinite,
            "total": pred.numel(),
        },
    )


def _coverage_failures(
    valid: Tensor,
    eligible: Tensor,
    units: Tensor,
    min_valid_fraction: float,
) -> list[Dict[str, Any]]:
    failures = []
    for unit in torch.unique(units):
        idx = (units == unit).nonzero(as_tuple=True)[0]
        eligible_count = int(eligible[idx].sum().item())
        valid_count = int(valid[idx].sum().item())
        coverage = valid_count / eligible_count if eligible_count else 0.0
        if valid_count == 0 or coverage < min_valid_fraction:
            failures.append(
                {
                    "unit": int(unit.item()),
                    "valid": valid_count,
                    "eligible": eligible_count,
                    "coverage": round(coverage, 8),
                }
            )
    return failures


def _require_target_coverage(
    valid: Tensor,
    eligible: Tensor,
    units: Tensor,
    min_valid_fraction: float,
    unit_name: str,
) -> None:
    failures = _coverage_failures(valid, eligible, units, min_valid_fraction)
    if failures:
        raise NumericalHealthError(
            "insufficient_target_coverage",
            {
                "unit_name": unit_name,
                "minimum": min_valid_fraction,
                "failures": failures,
            },
        )


class SILogLoss(nn.Module):
    """
    Scale-invariant logarithmic loss for monocular depth.

    L = alpha * sqrt(mean(d^2) - lambda * mean(d)^2)
    where d = log(pred) - log(gt), alpha=10, lambda=0.85.

    Reference: Eigen et al. 2014; DepthAnything v2 training.
    """

    def __init__(self, alpha: float = 10.0, lam: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred: (B, 1, H, W) predicted depth (positive).
            gt:   (B, 1, H, W) ground truth depth.
            mask: (B, 1, H, W) bool; valid pixels. None → use gt > 0.
        """
        pred = pred.clamp(min=self.eps)
        gt = gt.clamp(min=self.eps)

        if mask is None:
            mask = gt > 0

        d = torch.log(pred[mask]) - torch.log(gt[mask])
        if d.numel() == 0:
            return pred.sum() * 0.0  # differentiable zero

        var = d.pow(2).mean() - self.lam * d.mean().pow(2)
        return self.alpha * var.clamp(min=0.0).sqrt()


class TotalRetrainLoss(nn.Module):
    """
    Combined depth + distillation loss for the retrain phase.

    L_total = L_silog + lambda_distill * L_distill
    """

    def __init__(
        self,
        lambda_distill: float = 0.1,
        lambda_boundary: float = 1.0,
        lambda_scale: float = 1.0,
        silog_alpha: float = 10.0,
        silog_lam: float = 0.85,
    ):
        super().__init__()
        self.silog = SILogLoss(alpha=silog_alpha, lam=silog_lam)
        self.lambda_distill = lambda_distill
        self.lambda_boundary = lambda_boundary
        self.lambda_scale = lambda_scale

    def forward(
        self,
        student_out: Dict,
        gt_depth: Tensor,
        F_TA: Optional[Dict[str, Tensor]] = None,
        F_TB: Optional[Dict[str, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            student_out: dict from StudentNet.forward().
            gt_depth: (B, 1, H, W) ground truth depth.
            F_TA: boundary teacher features (optional; skip distill if None).
            F_TB: scale teacher features (optional).

        Returns:
            Dict with "total", "silog", "distill" losses.
        """
        depth_pred = student_out.get("depth")
        losses: Dict[str, Tensor] = {}

        if depth_pred is not None:
            if depth_pred.shape[2:] != gt_depth.shape[2:]:
                depth_pred = F.interpolate(
                    depth_pred, size=gt_depth.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            losses["silog"] = self.silog(depth_pred, gt_depth, mask)
        else:
            losses["silog"] = gt_depth.sum() * 0.0

        if F_TA is not None and F_TB is not None:
            losses["distill"] = dual_distill_loss(
                student_out, F_TA, F_TB,
                lambda_boundary=self.lambda_boundary,
                lambda_scale=self.lambda_scale,
            )
        else:
            device = losses["silog"].device
            losses["distill"] = torch.tensor(0.0, device=device)

        losses["total"] = losses["silog"] + self.lambda_distill * losses["distill"]
        return losses


class ScaleShiftInvariantDepthLoss(nn.Module):
    """Per-image normalized depth distillation loss.

    This is meant for monocular teachers such as DepthAnything whose prediction
    has reliable structure but arbitrary global scale and shift.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        lambda_grad: float = 0.5,
        min_valid_fraction: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self.eps = eps
        self.lambda_grad = lambda_grad
        self.min_valid_fraction = min_valid_fraction

    def forward(
        self,
        pred: Tensor,
        teacher: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if teacher.dim() == 3:
            teacher = teacher.unsqueeze(1)
        _require_finite_prediction(pred)
        pred = pred.float()
        teacher = teacher.float()
        if pred.shape[-2:] != teacher.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=teacher.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if pred.shape != teacher.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target "
                f"shape {tuple(teacher.shape)}"
            )
        _require_finite_prediction(pred)

        eligible = torch.ones_like(teacher, dtype=torch.bool)
        if mask is not None:
            eligible = _as_4d_mask(mask, teacher)
        valid = torch.isfinite(teacher) & eligible
        image_ids = torch.arange(pred.shape[0], device=pred.device)
        _require_target_coverage(
            valid,
            eligible,
            image_ids,
            self.min_valid_fraction,
            "image",
        )

        pred_norm = self._normalize(pred, valid)
        teacher_norm = self._normalize(teacher, valid).detach()

        pixel = self._masked_l1(pred_norm, teacher_norm, valid)
        grad = self._gradient_l1(pred_norm, teacher_norm, valid)
        total = pixel + self.lambda_grad * grad
        return {"total": total, "pixel": pixel, "grad": grad}

    def _normalize(self, x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.to(dtype=x.dtype)
        safe_x = torch.where(mask, x, torch.zeros_like(x))
        count = mask_f.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
        mean = (safe_x * mask_f).flatten(1).sum(dim=1).view(-1, 1, 1, 1) / count
        centered = safe_x - mean
        var = (centered.square() * mask_f).flatten(1).sum(dim=1).view(-1, 1, 1, 1) / count
        normalized = centered / var.add(self.eps).sqrt()
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def _masked_l1(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        values = (pred - target).abs()[mask]
        if values.numel() == 0:
            return pred.sum() * 0.0
        return values.mean()

    def _gradient_l1(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        mask_dx = mask[..., :, 1:] & mask[..., :, :-1]

        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        mask_dy = mask[..., 1:, :] & mask[..., :-1, :]

        loss_dx = self._masked_l1(pred_dx, target_dx, mask_dx)
        loss_dy = self._masked_l1(pred_dy, target_dy, mask_dy)
        return 0.5 * (loss_dx + loss_dy)


class MiDaSSSILoss(nn.Module):
    """MiDaS-style scale-shift-invariant distillation loss.

    Improvements over :class:`ScaleShiftInvariantDepthLoss` for distilling a
    relative monocular teacher (DepthAnything):

    * **Robust normalisation** — subtract per-image median, divide by mean
      absolute deviation (MAD) instead of z-score; far less sensitive to a few
      near/far outlier pixels that otherwise flatten the prediction.
    * **Trimmed data term** — drop the largest ``trim`` fraction of residuals
      (MiDaS ssitrim), so a handful of hard pixels don't dominate.
    * **Multi-scale gradient matching** — match residual gradients at
      ``num_scales`` resolutions; this is what actually preserves sharp object
      boundaries (a single-scale gradient term tends to oversmooth).

    Returns ``{"total", "pixel", "grad"}`` so it is a drop-in for the old loss's
    logging keys.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        lambda_grad: float = 1.0,
        num_scales: int = 4,
        trim: float = 0.2,
        min_valid_fraction: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self.eps = eps
        self.lambda_grad = lambda_grad
        self.num_scales = num_scales
        self.trim = trim
        self.min_valid_fraction = min_valid_fraction

    def forward(
        self,
        pred: Tensor,
        teacher: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if teacher.dim() == 3:
            teacher = teacher.unsqueeze(1)
        _require_finite_prediction(pred)
        # Cast before resizing so interpolation is also outside low-precision AMP.
        pred = pred.float()
        teacher = teacher.float()
        if pred.shape[-2:] != teacher.shape[-2:]:
            pred = F.interpolate(
                pred, size=teacher.shape[-2:], mode="bilinear", align_corners=False
            )
        if pred.shape != teacher.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target "
                f"shape {tuple(teacher.shape)}"
            )
        _require_finite_prediction(pred)

        eligible = torch.ones_like(teacher, dtype=torch.bool)
        if mask is not None:
            eligible = _as_4d_mask(mask, teacher)
        valid = torch.isfinite(teacher) & eligible
        image_ids = torch.arange(pred.shape[0], device=pred.device)
        _require_target_coverage(
            valid,
            eligible,
            image_ids,
            self.min_valid_fraction,
            "image",
        )

        pred_n = self._normalize(pred, valid)
        teacher_n = self._normalize(teacher, valid).detach()

        residual = pred_n - teacher_n
        # Invalid pseudo-target pixels do not participate in either data term.
        residual = torch.where(valid, residual, residual.new_zeros(()))
        pixel = self._trimmed_l1(residual, valid)
        grad = self._multiscale_grad(residual, valid)
        total = pixel + self.lambda_grad * grad
        return {"total": total, "pixel": pixel, "grad": grad}

    def _normalize(self, x: Tensor, mask: Tensor) -> Tensor:
        # Per-image median + mean-absolute-deviation over the valid pixels.
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        m_flat = mask.reshape(B, -1)
        med = x.new_zeros(B)
        scale = x.new_ones(B)
        for i in range(B):
            v = x_flat[i][m_flat[i]]
            if v.numel() == 0:
                continue
            mi = v.median()
            med[i] = mi
            scale[i] = (v - mi).abs().mean().clamp_min(self.eps)
        med = med.view(B, 1, 1, 1)
        scale = scale.view(B, 1, 1, 1)
        safe_x = torch.where(mask, x, med)
        normed = (safe_x - med) / scale
        return torch.where(mask, normed, torch.zeros_like(normed))

    def _trimmed_l1(self, residual: Tensor, mask: Tensor) -> Tensor:
        vals = residual.abs()[mask]
        if vals.numel() == 0:
            return residual.sum() * 0.0
        if self.trim > 0:
            keep = int(vals.numel() * (1.0 - self.trim))
            if keep > 0:
                vals, _ = torch.sort(vals)
                vals = vals[:keep]
        return vals.mean()

    def _multiscale_grad(self, residual: Tensor, mask: Tensor) -> Tensor:
        total = residual.sum() * 0.0
        r, m = residual, mask
        for _ in range(self.num_scales):
            total = total + self._grad_l1(r, m)
            if r.shape[-1] < 4 or r.shape[-2] < 4:
                break
            r = r[:, :, ::2, ::2]
            m = m[:, :, ::2, ::2]
        return total / self.num_scales

    @staticmethod
    def _grad_l1(r: Tensor, m: Tensor) -> Tensor:
        dx = (r[..., :, 1:] - r[..., :, :-1]).abs()
        mx = m[..., :, 1:] & m[..., :, :-1]
        dy = (r[..., 1:, :] - r[..., :-1, :]).abs()
        my = m[..., 1:, :] & m[..., :-1, :]
        loss = r.sum() * 0.0
        if mx.any():
            loss = loss + dx[mx].mean()
        if my.any():
            loss = loss + dy[my].mean()
        return 0.5 * loss


class GroupSSILoss(nn.Module):
    """Per-SCENE (group) scale-shift-invariant distillation loss.

    The whole point of multi-view-consistent pseudo-depth is that the per-frame
    SCALES are mutually consistent. :class:`MiDaSSSILoss` normalises every frame
    INDEPENDENTLY (per-image median/MAD), which erases exactly that information —
    ``normalize(a_f * depth_f) == normalize(depth_f)`` — so it cannot reward
    cross-frame consistency at all.

    This loss instead computes ONE median + MAD jointly over all frames of a
    group (a scene's sampled frames), and normalises every frame in the group by
    that single (m, s). A single global scale+shift for the whole scene is still
    factored out (DepthSplat works in a per-scene gauge anyway), but the RELATIVE
    scales between frames now survive into the residual — so a student that gets
    the cross-frame relative depth right is rewarded, and the consistency signal
    reaches the gradient. Reuses MiDaS trimmed-L1 + multi-scale gradient.

    forward(pred, teacher, groups, mask=None):
      pred, teacher: (N, 1, H, W)
      groups:        (N,) long; samples sharing a value are one scene/group.
    Returns {"total","pixel","grad"} for drop-in logging.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        lambda_grad: float = 1.0,
        num_scales: int = 4,
        trim: float = 0.2,
        min_valid_fraction: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self.eps = eps
        self.lambda_grad = lambda_grad
        self.num_scales = num_scales
        self.trim = trim
        self.min_valid_fraction = min_valid_fraction
        # Reuse MiDaS pixel/grad terms verbatim.
        self._midas = MiDaSSSILoss(
            eps, lambda_grad, num_scales, trim, min_valid_fraction
        )

    def forward(self, pred: Tensor, teacher: Tensor,
                groups: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if teacher.dim() == 3:
            teacher = teacher.unsqueeze(1)
        _require_finite_prediction(pred)
        # Cast before resizing so interpolation is also outside low-precision AMP.
        pred = pred.float()
        teacher = teacher.float()
        if pred.shape[-2:] != teacher.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=teacher.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if pred.shape != teacher.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target "
                f"shape {tuple(teacher.shape)}"
            )
        _require_finite_prediction(pred)
        if groups.dim() != 1 or groups.numel() != pred.shape[0]:
            raise ValueError(
                f"groups must have shape ({pred.shape[0]},), got {tuple(groups.shape)}"
            )

        groups = groups.to(device=pred.device)

        eligible = torch.ones_like(teacher, dtype=torch.bool)
        if mask is not None:
            eligible = _as_4d_mask(mask, teacher)
        valid = torch.isfinite(teacher) & eligible
        _require_target_coverage(
            valid,
            eligible,
            groups,
            self.min_valid_fraction,
            "group",
        )

        pred_n = self._group_normalize(pred, valid, groups)
        teacher_n = self._group_normalize(teacher, valid, groups).detach()

        residual = pred_n - teacher_n
        residual = torch.where(valid, residual, residual.new_zeros(()))
        pixel = self._midas._trimmed_l1(residual, valid)
        grad = self._midas._multiscale_grad(residual, valid)
        total = pixel + self.lambda_grad * grad
        return {"total": total, "pixel": pixel, "grad": grad}

    def _group_normalize(self, x: Tensor, mask: Tensor, groups: Tensor) -> Tensor:
        """One finite (median, MAD) per group, broadcast to its frames."""
        out = torch.zeros_like(x)
        for g in torch.unique(groups):
            idx = (groups == g).nonzero(as_tuple=True)[0]
            xs = x[idx]
            ms = mask[idx]
            v = xs[ms]
            # Coverage validation in forward guarantees this branch is non-empty.
            if v.numel() == 0:
                raise NumericalHealthError(
                    "empty_group_after_coverage_check",
                    {"group": int(g.item())},
                )
            med = v.median()
            scale = (v - med).abs().mean().clamp_min(self.eps)
            safe_xs = torch.where(ms, xs, med)
            normed = (safe_xs - med) / scale
            out[idx] = torch.where(ms, normed, torch.zeros_like(normed))
        return out
