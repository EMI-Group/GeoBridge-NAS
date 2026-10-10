"""Scale teacher: ResNet-34 + NLSPN-style UNet decoder, trained on TartanAir.

Architecture (NLSPN ECCV 2020 style):
  Encoder: ResNet-34 (ImageNet pretrained, first conv adapted to 4-ch input)
  Decoder: 4-stage top-down UNet with skip connections
  Input: RGB (3ch) + sparse depth (1ch) → 4ch
  Output: dense metric depth + 4 intermediate stage feature maps

Train with scripts/train_teacher.py on TartanAir before NAS search.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as tvm


SCALE_STAGE_KEYS = ["stage_0", "stage_1", "stage_2", "stage_3"]


# ── ResNet-34 backbone ────────────────────────────────────────────────────────

class ResNet34Backbone(nn.Module):
    """
    ResNet-34 adapted for 4-channel input (RGB + sparse depth).

    layer0 → stride 4,  64ch
    layer1 → stride 4,  64ch
    layer2 → stride 8,  128ch
    layer3 → stride 16, 256ch
    layer4 → stride 32, 512ch
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 4):
        super().__init__()
        base = tvm.resnet34(
            weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels, 64,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:]  = old_conv.weight.mean(1, keepdim=True)

        self.layer0 = nn.Sequential(new_conv, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 64ch,  stride 4
        self.layer2 = base.layer2   # 128ch, stride 8
        self.layer3 = base.layer3   # 256ch, stride 16
        self.layer4 = base.layer4   # 512ch, stride 32

    def forward(self, x: Tensor) -> List[Tensor]:
        x  = self.layer0(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [s1, s2, s3, s4]


# ── NLSPN-style UNet decoder ──────────────────────────────────────────────────

class NLSPNDecoder(nn.Module):
    """
    4-stage top-down decoder with skip connections.

    stage_0 : s4 upsamp → 256ch  at stride 16
    stage_1 : fuse s3   → 128ch  at stride 8
    stage_2 : fuse s2   → 64ch   at stride 4
    stage_3 : fuse s1   → 64ch   at stride 2
    depth   : 1ch full-res (upsampled in ScaleTeacher.forward)
    """

    def __init__(self, enc_channels: List[int] = [64, 128, 256, 512]):
        super().__init__()
        c0, c1, c2, c3 = enc_channels

        # After up3: d3 has c2 channels; skip s3 also c2 → cat = c2*2
        # After up2: d2 has c1 channels; skip s2 also c1 → cat = c1*2
        # After up1: d1 has c0 channels; skip s1 also c0 → cat = c0*2
        self.up3 = self._block(c3,       c2)        # 512→256
        self.up2 = self._block(c2 * 2,   c1)        # 512→128
        self.up1 = self._block(c1 * 2,   c0)        # 256→64
        self.up0 = self._block(c0 * 2,   c0)        # 128→64

        self.head = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0 // 2),
            nn.GELU(),
            nn.Conv2d(c0 // 2, 1, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, enc: List[Tensor]) -> Dict[str, Tensor]:
        s1, s2, s3, s4 = enc

        d3 = self.up3(s4)                            # stride 16
        d2 = self.up2(torch.cat([d3, s3], 1))        # stride 8
        d1 = self.up1(torch.cat([d2, s2], 1))        # stride 4
        d0 = self.up0(torch.cat([d1, s1], 1))        # stride 2

        return {
            "stage_0":   d3,
            "stage_1":   d2,
            "stage_2":   d1,
            "stage_3":   d0,
            "depth_raw": self.head(d0),
        }


# ── ScaleTeacher ──────────────────────────────────────────────────────────────

class ScaleTeacher(nn.Module):
    """
    NLSPN-based scale teacher (F_TB).

    After training on TartanAir, freeze and use for NAS search.

    Args:
        checkpoint_path:       pre-trained weights (.pth).  None → random init.
        device:                torch device.
        freeze:                freeze all params after loading.
        pretrained_backbone:   use ImageNet ResNet-34 backbone.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        freeze: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.encoder = ResNet34Backbone(pretrained=pretrained_backbone, in_channels=4)
        self.decoder = NLSPNDecoder(enc_channels=[64, 128, 256, 512])

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, device)
            print(f"[ScaleTeacher] Loaded checkpoint: {checkpoint_path}")

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

        self.to(device)
        if freeze:
            self.eval()

    def load_checkpoint(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device)
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[ScaleTeacher] Missing ({len(missing)}): {missing[:3]}…")
        if unexpected:
            print(f"[ScaleTeacher] Unexpected ({len(unexpected)}): {unexpected[:3]}…")

    @torch.no_grad()
    def extract_features(
        self,
        images: Tensor,
        sparse_depth: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass returning intermediate features + depth.

        Args:
            images:       (B, 3, H, W).
            sparse_depth: (B, 1, H, W) simulated LiDAR depth.

        Returns:
            Dict with "stage_0"…"stage_3" and "depth" (full resolution).
        """
        self.eval()
        x = torch.cat([images, sparse_depth], dim=1)   # (B, 4, H, W)
        enc  = self.encoder(x)
        dec  = self.decoder(enc)

        H, W = images.shape[2:]
        depth = F.interpolate(
            dec["depth_raw"], size=(H, W), mode="bilinear", align_corners=False
        )
        return {
            "stage_0": dec["stage_0"],
            "stage_1": dec["stage_1"],
            "stage_2": dec["stage_2"],
            "stage_3": dec["stage_3"],
            "depth":   depth,
        }

    def forward(self, images: Tensor, sparse_depth: Tensor) -> Dict[str, Tensor]:
        x = torch.cat([images, sparse_depth], dim=1)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        H, W = images.shape[2:]
        depth = F.interpolate(
            dec["depth_raw"], size=(H, W), mode="bilinear", align_corners=False
        )
        return {
            "stage_0": dec["stage_0"],
            "stage_1": dec["stage_1"],
            "stage_2": dec["stage_2"],
            "stage_3": dec["stage_3"],
            "depth":   depth,
        }

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()
