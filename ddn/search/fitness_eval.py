"""Fitness evaluation pipeline: builds StudentNet and runs grad-stability scoring.

Key design: the DINOv2 encoder is frozen and identical for every architecture.
It is therefore loaded ONCE and shared across all evaluations.  Encoder-level
features are reused.  Per-architecture work is limited to materializing the
fixed decoder trunk with candidate tree-coded Mamba cells and running one small
backward pass for gradient stability.
"""

import gc
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..search_space.tree import DecoderTree
from ..search_space.tree import make_fpn_template
from ..student.student_net import StudentNet
from ..proxy.grad_stability import ArchScore, grad_stability_objective
from ..proxy.grad_sensitivity import compute_grad_sensitivity
from ..proxy.distill_loss import SHALLOW_LAYERS, DEEP_LAYERS


class FitnessEvaluator:
    """
    Evaluates gradient-stability fitness for a candidate tree-coded Mamba decoder.

    The DINOv2 encoder is loaded **once** at construction time and reused for
    every architecture.  Per-architecture cost is dominated by a single small
    backward pass through the decoder for gradient stability.

    Args:
        batch:              (B, 3, H, W) image batch.
        F_TA:               boundary teacher features {layer_name: (B, D) Tensor}.
        F_TB:               scale teacher features {layer_name: (B, D) Tensor}.
        device:             torch device string.
        decoder_channels:   channel width inside the fixed Mamba decoder trunk.
        forward_batch_size: images per encoder forward pass (caps VRAM; default 4).
        grad_batch_size:    images for the grad-sensitivity backward pass (default 8).
        dinov2_weight_path: explicit path to DINOv2 .pth (optional).
    """

    def __init__(
        self,
        batch: Tensor,
        F_TA: Dict[str, Tensor],
        F_TB: Dict[str, Tensor],
        device: str = "cuda",
        decoder_channels: int = 256,
        forward_batch_size: int = 4,
        grad_batch_size: int = 8,
        dinov2_weight_path: Optional[str] = None,
        depth_target: Optional[Tensor] = None,
    ):
        self.device = device
        self.decoder_channels = decoder_channels
        self.forward_batch_size = forward_batch_size
        self.grad_batch_size = grad_batch_size

        n_grad = min(grad_batch_size, len(batch))
        self.batch = batch[:n_grad].to(device)
        self.F_TA  = {k: v[:n_grad].to(device) for k, v in F_TA.items()}
        self.F_TB  = {k: v[:n_grad].to(device) for k, v in F_TB.items()}

        # ── Load encoder ONCE ────────────────────────────────────────────────
        self._net = StudentNet(
            decoder_tree=make_fpn_template(decoder_channels),
            trainable_bias=False,
            decoder_channels=decoder_channels,
            weight_path=dinov2_weight_path,
        ).to(device).eval()

        # Grad sub-batch (small for backward pass)
        self._grad_batch = self.batch
        self._grad_F_TA  = self.F_TA
        self._grad_F_TB  = self.F_TB
        self._depth_target = (
            depth_target[:n_grad].to(device) if depth_target is not None else None
        )

    # ── Per-architecture evaluation ──────────────────────────────────────────

    def __call__(self, arch: DecoderTree, generation: int = 0) -> ArchScore:
        """
        Evaluate one candidate architecture.

        The score aggregates three independent decoder initializations,
        matching the evaluation rule reported in the paper: each
        initialization yields one raw forward--backward response and the
        candidate's response is their arithmetic mean (the log transform is
        applied afterwards by grad_stability_objective).

        Encoder is reused; the decoder is re-instantiated per initialization.
        Grad-sensitivity runs a backward pass through the decoder only.
        """
        raw_scores = []
        for _ in range(3):
            new_decoder = arch.materialize().to(self.device)
            old_decoder = self._net.decoder
            self._net.decoder = new_decoder

            raw_scores.append(compute_grad_sensitivity(
                self._net,
                self._grad_batch,
                self._grad_F_TA,
                self._grad_F_TB,
                SHALLOW_LAYERS,
                DEEP_LAYERS,
                depth_target=self._depth_target,
            ))

            self._net.decoder = old_decoder
            del new_decoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        score = ArchScore(arch_id=id(arch))
        score.score_grad = sum(raw_scores) / len(raw_scores)
        score.score_grad_stability = grad_stability_objective(score.score_grad)
        score.flops  = float(arch.flops(self.batch.shape[2], self.batch.shape[3]))
        score.params = float(arch.params())

        return score


def build_evaluator_from_cache(
    cache_dir: str,
    batch_indices: List[int],
    device: str = "cuda",
    decoder_channels: int = 256,
    forward_batch_size: int = 4,
    grad_batch_size: int = 8,
    dinov2_weight_path: Optional[str] = None,
    depth_target_path: Optional[str] = None,
) -> FitnessEvaluator:
    """
    Build a FitnessEvaluator from a pre-computed LMDB feature cache.

    Args:
        cache_dir:      LMDB cache path written by scripts/cache_features.py.
        batch_indices:  sample indices to form the evaluation batch.
    """
    from ..data.feature_cache import FeatureCache

    B = len(batch_indices)
    assert B >= grad_batch_size, (
        f"Need at least {grad_batch_size} samples, got {B}"
    )

    with FeatureCache(cache_dir, write=False) as cache:
        ta_feats: Dict[str, List[Tensor]] = {l: [] for l in SHALLOW_LAYERS}
        tb_feats: Dict[str, List[Tensor]] = {l: [] for l in DEEP_LAYERS}
        images: List[Tensor] = []

        for idx in batch_indices:
            entry = cache.get(idx)
            assert entry is not None, f"Cache miss for idx={idx}"
            images.append(entry["image"])
            for l in SHALLOW_LAYERS:
                ta_feats[l].append(entry["F_TA"][l])
            for l in DEEP_LAYERS:
                tb_feats[l].append(entry["F_TB"][l])

    batch = torch.stack(images)
    F_TA  = {l: torch.stack(ta_feats[l]) for l in SHALLOW_LAYERS}
    F_TB  = {l: torch.stack(tb_feats[l]) for l in DEEP_LAYERS}

    depth_target = None
    if depth_target_path:
        payload = torch.load(depth_target_path, map_location="cpu")
        expected_indices = list(payload.get("cache_indices", []))
        if expected_indices and expected_indices != list(batch_indices):
            raise ValueError(
                "Spatial depth target cache_indices do not match the evaluator batch_indices: "
                f"{expected_indices} != {list(batch_indices)}"
            )
        depth_target = payload["target"].float()
    return FitnessEvaluator(
        batch=batch,
        F_TA=F_TA,
        F_TB=F_TB,
        device=device,
        decoder_channels=decoder_channels,
        forward_batch_size=forward_batch_size,
        grad_batch_size=grad_batch_size,
        dinov2_weight_path=dinov2_weight_path,
        depth_target=depth_target,
    )
