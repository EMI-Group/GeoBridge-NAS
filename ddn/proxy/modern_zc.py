"""Recent zero-cost proxies requested by review: ZiCo, SynFlow and AZ-NAS.

All three are scored on the *decoder only*, with the same cached-feature loss
that GSAP uses, so the comparison isolates the proxy and not the signal source.

Why these three:

* **ZiCo** (Li et al., ICLR'23) is the standard modern gradient proxy and, unlike
  GSAP, is explicitly designed to be robust to network width -- it aggregates the
  inverse coefficient of variation of gradients across a batch, which is scale-
  free per parameter tensor. It is the fair "does a modern proxy already solve
  the size-coupling problem?" control.
* **SynFlow** (Tanaka et al., NeurIPS'20) is data independent and is *known* to
  be strongly size-biased. It is included as the opposite pole, so the size
  analysis has a reference point at both ends.
* **AZ-NAS** (Lee & Ham, CVPR'24) is the modern *aggregator* the review asks for:
  it combines expressivity / progressivity / trainability / complexity by rank
  aggregation rather than by a single scalar.

ParZC and TG-NAS are deliberately not implemented here: both require a predictor
pretrained on cell-based classification benchmarks (NAS-Bench-201/301), and
transferring them to this decoder space would mean retraining the predictor on
our own fully-trained pool -- which would leak the very labels the proxy study
is meant to predict.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .distill_loss import SHALLOW_LAYERS, DEEP_LAYERS

ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU, nn.Hardswish)


def _cached_target_loss(student, batch, F_TA, shallow_layers=SHALLOW_LAYERS):
    """The same search-time loss as GSAP: MSE to the channel mean of f_a."""
    out = (student.forward_decoder_only(batch)
           if hasattr(student, "forward_decoder_only") else student(batch))
    depth = out["depth"]
    t = F_TA[shallow_layers[0]]
    target = (t.mean(dim=1, keepdim=True)
              .unsqueeze(-1).unsqueeze(-1).expand_as(depth).detach())
    return F.mse_loss(depth, target), out


# ── ZiCo ────────────────────────────────────────────────────────────────────

def compute_zico(
    student: nn.Module,
    batch: Tensor,
    F_TA: Dict[str, Tensor],
    F_TB: Dict[str, Tensor],
    n_splits: int = 4,
    decoder_attr: str = "decoder",
) -> float:
    """ZiCo: sum over parameter tensors of log( sum_i |mean_b g_i| / std_b g_i ).

    The batch is split into `n_splits` micro-batches to obtain the across-batch
    gradient mean and standard deviation. Higher is better.
    """
    decoder = getattr(student, decoder_attr)
    was_training = student.training
    student.train()

    chunks = torch.chunk(batch, n_splits, dim=0)
    fa_chunks = {k: torch.chunk(v, n_splits, dim=0) for k, v in F_TA.items()}

    grads: List[List[Tensor]] = []
    for ci, chunk in enumerate(chunks):
        if chunk.shape[0] == 0:
            continue
        student.zero_grad(set_to_none=True)
        fa = {k: v[ci] for k, v in fa_chunks.items()}
        loss, _ = _cached_target_loss(student, chunk, fa)
        if not loss.requires_grad:
            continue
        loss.backward()
        grads.append([
            (p.grad.detach().clone().double() if p.grad is not None
             else torch.zeros_like(p, dtype=torch.float64))
            for p in decoder.parameters()
        ])

    if not was_training:
        student.eval()
    if len(grads) < 2:
        return float("nan")

    score = 0.0
    for j in range(len(grads[0])):
        stack = torch.stack([g[j] for g in grads], dim=0)   # (splits, ...)
        mean = stack.mean(0).abs()
        std = stack.std(0)
        nz = std > 0
        if nz.any():
            val = float((mean[nz] / std[nz]).sum())
            if val > 0:
                score += float(torch.log(torch.tensor(val)))
    return score


# ── SynFlow ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _linearize(model):
    signs = {}
    for name, p in model.state_dict().items():
        signs[name] = torch.sign(p)
        p.abs_()
    return signs


@torch.no_grad()
def _restore(model, signs):
    for name, p in model.state_dict().items():
        if name in signs:
            p.mul_(signs[name])


def compute_synflow(
    student: nn.Module,
    batch: Tensor,
    F_TA: Dict[str, Tensor],
    F_TB: Dict[str, Tensor],
    decoder_attr: str = "decoder",
) -> float:
    """SynFlow: data-independent sum of |param * grad| under an all-ones input.

    Weights are temporarily made positive and the loss is the sum of the output,
    so the score depends only on topology and parameter magnitudes.
    """
    decoder = getattr(student, decoder_attr)
    was_training = student.training
    student.train()
    signs = _linearize(student)
    student.zero_grad(set_to_none=True)

    try:
        ones = torch.ones_like(batch)
        out = (student.forward_decoder_only(ones)
               if hasattr(student, "forward_decoder_only") else student(ones))
        depth = out["depth"]
        if not depth.requires_grad:
            return float("nan")
        depth.sum().backward()
        score = 0.0
        for p in decoder.parameters():
            if p.grad is not None:
                score += float((p.detach() * p.grad.detach()).abs().sum())
    finally:
        _restore(student, signs)
        student.zero_grad(set_to_none=True)
        if not was_training:
            student.eval()
    return score


# ── AZ-NAS ──────────────────────────────────────────────────────────────────

def compute_aznas(
    student: nn.Module,
    batch: Tensor,
    F_TA: Dict[str, Tensor],
    F_TB: Dict[str, Tensor],
    decoder_attr: str = "decoder",
) -> Dict[str, float]:
    """AZ-NAS component scores (expressivity, progressivity, trainability, complexity).

    Returns the four raw components. AZ-NAS combines them by *rank* aggregation
    across the candidate population, so the ranking is done by the caller once
    every architecture has been scored -- see scripts/25.
    """
    decoder = getattr(student, decoder_attr)
    n = batch.shape[0]
    feats: List[Tensor] = []

    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(x) and x.dim() >= 2 and x.shape[0] == n:
            feats.append(x.detach())

    handles = [m.register_forward_hook(hook)
               for m in decoder.modules() if isinstance(m, ACT_TYPES)]

    was_training = student.training
    student.train()
    student.zero_grad(set_to_none=True)
    try:
        loss, _ = _cached_target_loss(student, batch, F_TA)
        # trainability needs the gradient; expressivity/progressivity use feats
        if loss.requires_grad:
            loss.backward()
    finally:
        for h in handles:
            h.remove()

    # expressivity: log-determinant style diversity of the final feature map
    expressivity = 0.0
    if feats:
        x = feats[-1].reshape(n, -1).double()
        x = x - x.mean(0, keepdim=True)
        cov = (x @ x.t()) / max(1, x.shape[1])
        cov += 1e-4 * torch.eye(n, dtype=cov.dtype, device=cov.device) * cov.diag().mean().clamp(min=1e-12)
        _s, logdet = torch.slogdet(cov)
        expressivity = float(logdet)

    # progressivity: the minimum per-stage increase in feature diversity
    progressivity = 0.0
    if len(feats) >= 2:
        divs = []
        for fmap in feats:
            x = fmap.reshape(n, -1).double()
            x = x - x.mean(0, keepdim=True)
            divs.append(float(torch.log(x.pow(2).mean().clamp(min=1e-12))))
        progressivity = float(min(b - a for a, b in zip(divs[:-1], divs[1:])))

    # trainability: gradient-flow stability across parameter tensors
    ratios = []
    for p in decoder.parameters():
        if p.grad is not None and p.numel() > 0:
            gn = float(p.grad.detach().norm())
            pn = float(p.detach().norm())
            if pn > 0 and gn > 0:
                ratios.append(gn / pn)
    trainability = 0.0
    if ratios:
        t = torch.tensor(ratios, dtype=torch.float64)
        # penalize dispersion of the gradient-to-weight ratio (AZ-NAS favors
        # uniform gradient flow across the network)
        trainability = float(-torch.log(t.std() / t.mean().clamp(min=1e-12) + 1e-12))

    complexity = float(sum(p.numel() for p in decoder.parameters()))

    student.zero_grad(set_to_none=True)
    if not was_training:
        student.eval()

    return {"expressivity": expressivity, "progressivity": progressivity,
            "trainability": trainability, "complexity": complexity}
