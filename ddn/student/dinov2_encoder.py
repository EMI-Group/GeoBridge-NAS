"""DINOv2 ViT-S encoder with bias-only fine-tuning support.

Uses local vendor copy of the DINOv2 implementation (no transformers needed).
Weights: dinov2_vits14_pretrain.pth  from torch.hub cache or checkpoints/.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

_VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendors" / "depth_anything_v2"

INTERMEDIATE_LAYERS = [2, 5, 8, 11]    # ViT-S has 12 transformer blocks
FEAT_DIM  = 384                         # ViT-S embed_dim
PATCH_SIZE = 14                         # DINOv2 patch size


def _build_dinov2_vits() -> nn.Module:
    from ddn.vendors.depth_anything_v2.dinov2 import DINOv2 as _Factory
    return _Factory("vits")


class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-S encoder with selective bias-only training.

    NAS search phase   – fully frozen.
    Full retrain phase – only bias params trainable (~0.1% of total).

    Args:
        pretrained:      load dinov2_vits14_pretrain.pth.
        trainable_bias:  unfreeze bias params (for retrain phase).
        layers:          ViT block indices to extract features from.
        weight_path:     explicit .pth path; falls back to torch.hub cache.
    """

    WEIGHT_URL       = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    WEIGHT_FILENAME  = "dinov2_vits14_pretrain.pth"

    def __init__(
        self,
        pretrained: bool = True,
        trainable_bias: bool = False,
        layers: List[int] = INTERMEDIATE_LAYERS,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.layers = layers
        self._model: nn.Module = _build_dinov2_vits()

        if pretrained:
            self._load_weights(weight_path)

        self._set_trainable(trainable_bias)

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_weights(self, explicit: Optional[str]) -> None:
        path = self._find_weights(explicit)
        if path is None:
            print(f"[DINOv2Encoder] Downloading weights…")
            state = torch.hub.load_state_dict_from_url(self.WEIGHT_URL, map_location="cpu")
        else:
            print(f"[DINOv2Encoder] Loading weights from {path}")
            state = torch.load(path, map_location="cpu")
        state = self._extract_backbone(state)
        n_target = len(self._model.state_dict())
        matched = sum(1 for k in state if k in self._model.state_dict())
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        print(f"[DINOv2Encoder] Loaded {matched}/{n_target} backbone tensors")
        # A path was given but nothing matched -> the encoder would be random.
        # That silently happened with raw Depth-Anything checkpoints (keys live
        # under a 'pretrained.' prefix) and is the kind of bug that wastes a full
        # training run, so fail loud instead.
        if matched == 0:
            raise RuntimeError(
                f"[DINOv2Encoder] 0 keys matched from {path}. Got sample keys "
                f"{list(state)[:3]}; expected DINOv2 keys like "
                f"{list(self._model.state_dict())[:3]}."
            )
        if missing:
            print(f"[DINOv2Encoder] Missing ({len(missing)}): {missing[:2]}…")
        if unexpected:
            print(f"[DINOv2Encoder] Unexpected ({len(unexpected)}): {unexpected[:2]}…")

    @staticmethod
    def _extract_backbone(state: dict) -> dict:
        """Normalise a checkpoint to bare DINOv2 backbone keys.

        Accepts raw DINOv2 SSL checkpoints AND Depth-Anything-V2 checkpoints
        (whose DINOv2 backbone lives under a ``pretrained.`` prefix alongside a
        ``depth_head.``). Lets ``--dinov2_ckpt`` point at either so the student
        encoder can be initialised from DA's depth-tuned backbone directly.
        """
        for wrap in ("model", "state_dict", "teacher", "student"):
            if isinstance(state, dict) and wrap in state and isinstance(state[wrap], dict):
                state = state[wrap]
        if any(k.startswith("pretrained.") for k in state):
            state = {
                k[len("pretrained."):]: v
                for k, v in state.items()
                if k.startswith("pretrained.")
            }
        return state

    def _find_weights(self, explicit: Optional[str]) -> Optional[str]:
        candidates = []
        if explicit:
            candidates.append(explicit)
        # project checkpoints/
        candidates.append(str(
            Path(__file__).resolve().parents[2] / "checkpoints" / self.WEIGHT_FILENAME
        ))
        # torch hub cache
        candidates.append(str(
            Path(torch.hub.get_dir()) / "checkpoints" / self.WEIGHT_FILENAME
        ))
        for p in candidates:
            if p and Path(p).exists():
                return p
        return None

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def _set_trainable(self, bias_only: bool) -> None:
        # When bias_only=True we run BitFit + LayerNorm + LayerScale: all
        # `*.bias`, all LayerNorm weights (`norm`/`norm1`/`norm2`), and all
        # LayerScale gammas (`ls1.gamma`/`ls2.gamma`) are trained, while the
        # heavy QKV/MLP/patch_embed weights stay frozen. This closes most of
        # the BitFit→full-finetune gap on KITTI for ~0.1% extra params.
        for name, p in self._model.named_parameters():
            if not bias_only:
                p.requires_grad_(False)
                continue
            is_bias = name.endswith(".bias") or name == "bias"
            is_ln = any(part.startswith("norm") for part in name.split("."))
            is_layerscale = name.endswith(".gamma")
            p.requires_grad_(is_bias or is_ln or is_layerscale)

    def enable_bias_training(self) -> None:
        """Unlock bias params before full retraining."""
        self._set_trainable(True)

    def enable_full_finetune(self) -> None:
        """Unlock all encoder params except ones unused by forward.

        ``mask_token`` is only used in masked pretraining; ``get_intermediate_layers``
        never touches it, so keep it frozen to avoid DDP "param did not receive
        grad" errors during full fine-tuning.
        """
        for name, p in self._model.named_parameters():
            p.requires_grad_("mask_token" not in name)

    def enable_partial_finetune(self, n_blocks: int) -> None:
        """Train only the last ``n_blocks`` transformer blocks (+ final norm),
        freezing patch_embed / pos_embed / cls_token / earlier blocks.

        A middle ground between BitFit (``n_blocks<=0``) and full fine-tune
        (``n_blocks>=depth``): adds real backbone capacity for high-redundancy
        data (e.g. DDAD video frames) without letting the whole encoder memorise
        the train set. ``mask_token`` stays frozen (unused by forward → keeps DDP
        find_unused_parameters=False valid); the final ``norm`` and the last
        ``n_blocks`` are all touched by ``get_intermediate_layers`` so no
        trainable param goes gradient-less.
        """
        depth = len(self._model.blocks)
        n = max(0, min(n_blocks, depth))
        if n == 0:
            self._set_trainable(True)        # fall back to BitFit
            return
        if n >= depth:
            self.enable_full_finetune()
            return
        for p in self._model.parameters():
            p.requires_grad_(False)
        for blk in list(self._model.blocks)[depth - n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if getattr(self._model, "norm", None) is not None:
            for p in self._model.norm.parameters():
                p.requires_grad_(True)

    def freeze_all(self) -> None:
        self._set_trainable(False)

    # ── Feature extraction ────────────────────────────────────────────────────

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Optional[List[int]] = None,
    ) -> List[Tensor]:
        """
        Extract patch-token features reshaped to spatial maps.

        Args:
            x: (B, 3, H, W).  H and W must be multiples of 14.
            n: block indices to extract (default: self.layers).

        Returns:
            List of (B, 1024, H_p, W_p) tensors, H_p = H // 14.
        """
        layer_ids = n if n is not None else self.layers
        B, _, H, W = x.shape
        H_p, W_p = H // PATCH_SIZE, W // PATCH_SIZE

        ctx = torch.enable_grad() if self.training else torch.no_grad()
        with ctx:
            raw = self._model.get_intermediate_layers(
                x, layer_ids, return_class_token=True
            )

        feats = []
        for patch_tokens, _cls in raw:        # patch_tokens: (B, N, D)
            feat = patch_tokens.reshape(B, H_p, W_p, FEAT_DIM)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(feat)
        return feats

    def forward(self, x: Tensor) -> Dict[str, object]:
        feats = self.get_intermediate_layers(x)
        return {
            "shallow_feats": feats[:2],   # blocks 4, 11
            "deep_feats":    feats[2:],   # blocks 17, 23
            "all_feats":     feats,
        }

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self._model.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_params())

    def count_total(self) -> int:
        return sum(p.numel() for p in self._model.parameters())
