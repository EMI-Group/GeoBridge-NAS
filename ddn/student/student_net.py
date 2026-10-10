"""StudentNet: DINOv2 ViT-S encoder + searched Mamba decoder."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dinov2_encoder import DINOv2Encoder, FEAT_DIM, INTERMEDIATE_LAYERS
from ..search_space.tree import DecoderTree, MaterializedDecoder, make_fpn_template
from ..vendors.external_mamba import ExternalMambaDecoder


class StudentNet(nn.Module):
    """
    Student network for DualDistill-NAS.

    Encoder: DINOv2 ViT-S (semi-frozen).
      NAS search   → fully frozen.
      Full retrain → only bias params trainable (~0.1% of total).

    Decoder: fixed DINO pyramid trunk with tree-coded Mamba cells.

    Args:
        decoder_tree:     searched Mamba-cell genotype; defaults to a random trunk.
        trainable_bias:   unlock encoder bias params (retrain phase).
        decoder_channels: feature channels inside the decoder.
        weight_path:      explicit path to DINOv2 ViT-S .pth.
        pretrained_encoder: load pretrained DINOv2 weights.
    """

    ENCODER_LAYERS = INTERMEDIATE_LAYERS   # [4, 11, 17, 23]

    def __init__(
        self,
        decoder_tree: Optional[DecoderTree] = None,
        trainable_bias: bool = False,
        decoder_channels: int = 256,
        weight_path: Optional[str] = None,
        pretrained_encoder: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        decoder_baseline: str = "tree",
    ):
        super().__init__()

        self.encoder = DINOv2Encoder(
            pretrained=pretrained_encoder,
            trainable_bias=trainable_bias,
            layers=self.ENCODER_LAYERS,
            weight_path=weight_path,
        )

        if decoder_baseline == "tree":
            if decoder_tree is None:
                decoder_tree = make_fpn_template(channels=decoder_channels)
            # Override the genotype's head bounds with the dataset-specific values
            # passed in here. Depth bounds are a training/dataset config, not part
            # of the architecture identity, so they're not stored in the genotype.
            decoder_tree.min_depth = float(min_depth)
            decoder_tree.max_depth = float(max_depth)
            self.decoder = decoder_tree.materialize()
        elif decoder_baseline in {"vmamba", "localmamba"}:
            self.decoder = ExternalMambaDecoder(
                decoder_baseline, decoder_channels, float(min_depth), float(max_depth)
            )
        else:
            raise ValueError(f"Unknown decoder_baseline={decoder_baseline}")

        # Project each DINOv2 ViT-S layer feature to the decoder width.  These
        # maps still share the same token-grid resolution; the decoder trunk
        # builds the explicit pyramid internally.
        self.scale_proj = nn.ModuleDict({
            f"s{s}": nn.Conv2d(FEAT_DIM, decoder_channels, 1, bias=False)
            for s in [32, 16, 8, 4]
        })
        self.decoder_channels = decoder_channels

    # ── Phase transitions ─────────────────────────────────────────────────────

    def enable_retrain_mode(self) -> None:
        """Unfreeze encoder biases for the full retraining phase."""
        self.encoder.enable_bias_training()

    def enable_full_finetune(self) -> None:
        """Unfreeze the entire encoder (full fine-tuning, not BitFit)."""
        self.encoder.enable_full_finetune()

    def enable_partial_finetune(self, n_blocks: int) -> None:
        """Unfreeze only the last ``n_blocks`` encoder blocks (+ final norm)."""
        self.encoder.enable_partial_finetune(n_blocks)

    def freeze_encoder(self) -> None:
        self.encoder.freeze_all()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Dict[str, object]:
        """
        Args:
            x: (B, 3, H, W) normalised input images.

        Returns:
            "depth":             (B, 1, H, W).
            "shallow_feats":     list[Tensor] — blocks 4 & 11 (boundary alignment).
            "deep_feats":        list[Tensor] — blocks 17 & 23 (scale alignment).
            "dec_intermediates": dict of decoder intermediate maps.
        """
        # DINOv2 patch size is 14; pad H and W to the next multiple of 14.
        _patch = 14
        orig_H, orig_W = x.shape[2], x.shape[3]
        pH = ((_patch - orig_H % _patch) % _patch)
        pW = ((_patch - orig_W % _patch) % _patch)
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH), mode="reflect")

        enc_feats = self.encoder.get_intermediate_layers(x, n=self.ENCODER_LAYERS)

        scales = [32, 16, 8, 4]
        projected = [
            self.scale_proj[f"s{s}"](enc_feats[i])
            for i, s in enumerate(scales)
            if i < len(enc_feats)
        ]

        dec_out = self.decoder(projected)

        depth = dec_out.get("depth")
        if depth is not None and depth.shape[2:] != (orig_H, orig_W):
            depth = F.interpolate(
                depth, size=(orig_H, orig_W), mode="bilinear", align_corners=False
            )

        return {
            "depth":             depth,
            "shallow_feats":     enc_feats[:2],
            "deep_feats":        enc_feats[2:],
            "dec_intermediates": dec_out,
        }

    def forward_decoder_only(self, x: Tensor) -> Dict[str, object]:
        """
        Like forward(), but runs the encoder under no_grad to avoid storing
        its huge computation graph. Only scale_proj + decoder get grad.
        Use this for grad-sensitivity computation to save VRAM.
        """
        _patch = 14
        orig_H, orig_W = x.shape[2], x.shape[3]
        pH = ((_patch - orig_H % _patch) % _patch)
        pW = ((_patch - orig_W % _patch) % _patch)
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH), mode="reflect")

        with torch.no_grad():
            enc_feats = self.encoder.get_intermediate_layers(x, n=self.ENCODER_LAYERS)

        scales = [32, 16, 8, 4]
        projected = [
            self.scale_proj[f"s{s}"](enc_feats[i])
            for i, s in enumerate(scales)
            if i < len(enc_feats)
        ]

        dec_out = self.decoder(projected)
        depth = dec_out.get("depth")
        if depth is not None and depth.shape[2:] != (orig_H, orig_W):
            depth = F.interpolate(
                depth, size=(orig_H, orig_W), mode="bilinear", align_corners=False
            )
        return {"depth": depth, "dec_intermediates": dec_out}

    # ── Parameter groups ──────────────────────────────────────────────────────

    def decoder_parameters(self) -> List[nn.Parameter]:
        return list(self.decoder.parameters()) + list(self.scale_proj.parameters())

    def encoder_bias_parameters(self) -> List[nn.Parameter]:
        return [
            p for n, p in self.encoder.named_parameters()
            if "bias" in n and p.requires_grad
        ]

    def encoder_finetune_parameters(self) -> List[nn.Parameter]:
        # Every encoder param that the encoder itself marked trainable
        # (BitFit / BitFit+LN+LayerScale / full finetune all funnel here).
        return [p for _, p in self.encoder.named_parameters() if p.requires_grad]

    def param_count(self) -> Dict[str, int]:
        enc_total = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_total = sum(p.numel() for p in self.decoder_parameters())
        return {
            "encoder_total":     enc_total,
            "encoder_trainable": enc_train,
            "decoder_total":     dec_total,
        }
