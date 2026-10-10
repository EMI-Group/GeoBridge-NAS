"""Boundary teacher: DepthAnything v2 ViT-L (frozen).

Uses local vendor implementation — no transformers dependency.
Weights: depth_anything_v2_vitl.pth in checkpoints/.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

_VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendors" / "depth_anything_v2"
_CKPT_DIR   = Path(__file__).resolve().parents[2] / "checkpoints"

BOUNDARY_LAYERS = [4, 11, 17, 23]
PATCH_SIZE = 14
FEAT_DIM   = 1024


def _build_depth_anything_vitl() -> nn.Module:
    from ddn.vendors.depth_anything_v2.dpt import DepthAnythingV2
    return DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )


class BoundaryTeacher(nn.Module):
    """
    Frozen DepthAnything v2 ViT-L used as boundary teacher (F_TA).

    Features are extracted from ViT intermediate blocks [4, 11, 17, 23]
    and returned as spatial maps (B, 1024, H_p, W_p).
    The model is permanently frozen; all extraction is done under no_grad.

    Args:
        pretrained:   load weights from checkpoints/depth_anything_v2_vitl.pth.
        layers:       ViT block indices to extract.
        weight_path:  explicit .pth path (overrides default search).
        device:       device string.
    """

    WEIGHT_FILENAME = "depth_anything_v2_vitl.pth"
    WEIGHT_URL = (
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large"
        "/resolve/main/depth_anything_v2_vitl.pth"
    )

    def __init__(
        self,
        pretrained: bool = True,
        layers: List[int] = BOUNDARY_LAYERS,
        weight_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.layers = layers
        self._model: nn.Module = _build_depth_anything_vitl()

        if pretrained:
            self._load_weights(weight_path)

        for p in self._model.parameters():
            p.requires_grad_(False)

        self._model = self._model.to(device)
        self._model.eval()

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_weights(self, explicit: Optional[str]) -> None:
        path = self._find_weights(explicit)
        if path is None:
            raise FileNotFoundError(
                f"[BoundaryTeacher] Cannot find {self.WEIGHT_FILENAME}.\n"
                f"Download: {self.WEIGHT_URL}\n"
                f"Save to:  {_CKPT_DIR}/"
            )
        print(f"[BoundaryTeacher] Loading weights from {path}")
        state = torch.load(path, map_location="cpu")
        self._model.load_state_dict(state, strict=True)

    def _find_weights(self, explicit: Optional[str]) -> Optional[str]:
        for p in [
            explicit,
            str(_CKPT_DIR / self.WEIGHT_FILENAME),
        ]:
            if p and Path(p).exists():
                return p
        return None

    # ── Feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Extract intermediate ViT features as spatial maps.

        Args:
            images: (B, 3, H, W) normalised ImageNet RGB.

        Returns:
            Dict "layer_4" / "layer_11" / "layer_17" / "layer_23"
            → (B, 1024, H_p, W_p), H_p = H_padded // 14.
        """
        self._model.eval()
        B, _, H, W = images.shape

        # DINOv2 requires H and W to be multiples of patch size (14).
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h))

        H_p, W_p = images.shape[2] // PATCH_SIZE, images.shape[3] // PATCH_SIZE

        raw = self._model.pretrained.get_intermediate_layers(
            images, self.layers, return_class_token=True
        )

        feats: Dict[str, Tensor] = {}
        for layer_idx, (patch_tokens, _cls) in zip(self.layers, raw):
            feat = patch_tokens.reshape(B, H_p, W_p, FEAT_DIM)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            feats[f"layer_{layer_idx}"] = feat
        return feats

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.extract_features(images)

    @torch.no_grad()
    def predict_depth(self, images: Tensor) -> Tensor:
        """Predict relative DepthAnything depth for normalized RGB images.

        DepthAnything/DINOv2 requires spatial sizes divisible by the patch
        size.  Training code can pass arbitrary resized batches; this helper
        pads them before inference and crops the prediction back afterwards.
        """
        self._model.eval()
        _, _, height, width = images.shape
        pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode="reflect")

        depth = self._model(images)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[-2:] != images.shape[-2:]:
            depth = torch.nn.functional.interpolate(
                depth,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        return depth[..., :height, :width]
