"""Traditional CNN/FPN decoder search spaces for search-space ablation.

These architectures intentionally keep the same frozen DINO encoder interface
as the Mamba-tree decoder, but replace the searched stage-local Mamba trees
with conventional CNN decoder primitives. This lets the paper compare search
spaces while keeping the evaluator, search budget and retraining script shared.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tree import PATCH_SIZE, STAGE_SCALES, _AdaBinsHead


REFINE_CHOICES = ("conv3", "sepconv3", "dilconv3", "resblock", "mbconv")
FUSION_CHOICES = ("add", "concat", "gated")
UPSAMPLE_CHOICES = ("bilinear", "deconv")


class SepConv3(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class MBConv(nn.Module):
    def __init__(self, channels: int, expand: int = 4):
        super().__init__()
        hidden = channels * expand
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


def make_refine(name: str, channels: int) -> nn.Module:
    if name == "conv3":
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
    if name == "sepconv3":
        return SepConv3(channels)
    if name == "dilconv3":
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
    if name == "resblock":
        return ResBlock(channels)
    if name == "mbconv":
        return MBConv(channels)
    raise ValueError(f"Unknown traditional refine op: {name}")


class Fusion(nn.Module):
    def __init__(self, mode: str, channels: int):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv2d(channels * 2, channels, 1, bias=False)
        elif mode == "gated":
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.Sigmoid(),
            )
        elif mode != "add":
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, top: Tensor, lateral: Tensor) -> Tensor:
        if top.shape[-2:] != lateral.shape[-2:]:
            top = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "add":
            return top + lateral
        if self.mode == "concat":
            return self.proj(torch.cat([top, lateral], dim=1))
        gate = self.gate(torch.cat([top, lateral], dim=1))
        return gate * top + (1.0 - gate) * lateral


class Up(nn.Module):
    def __init__(self, mode: str, channels: int):
        super().__init__()
        self.mode = mode
        self.deconv = (
            nn.ConvTranspose2d(channels, channels, 2, stride=2, bias=False)
            if mode == "deconv"
            else None
        )
        if mode not in UPSAMPLE_CHOICES:
            raise ValueError(f"Unknown upsample mode: {mode}")

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        if self.deconv is not None:
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


@dataclass
class TraditionalDecoderArch:
    """Flat conventional decoder search space.

    The macro skeleton is a standard top-down FPN/U-Net decoder. The searchable
    part is a flat vector of CNN refine ops, fusion modes and upsample modes,
    which represents a traditional hand-designed decoder NAS space.
    """

    channels: int = 256
    refine_ops: List[str] = field(default_factory=list)
    fusion_ops: List[str] = field(default_factory=list)
    upsample_ops: List[str] = field(default_factory=list)
    min_depth: float = 1e-3
    max_depth: float = 80.0

    @staticmethod
    def random(channels: int = 256) -> "TraditionalDecoderArch":
        return TraditionalDecoderArch(
            channels=channels,
            refine_ops=[random.choice(REFINE_CHOICES) for _ in STAGE_SCALES],
            fusion_ops=[random.choice(FUSION_CHOICES) for _ in range(3)],
            upsample_ops=[random.choice(UPSAMPLE_CHOICES) for _ in range(3)],
        )

    def validate(self) -> bool:
        return (
            len(self.refine_ops) == 4
            and len(self.fusion_ops) == 3
            and len(self.upsample_ops) == 3
            and all(x in REFINE_CHOICES for x in self.refine_ops)
            and all(x in FUSION_CHOICES for x in self.fusion_ops)
            and all(x in UPSAMPLE_CHOICES for x in self.upsample_ops)
        )

    def materialize(self) -> "TraditionalDecoder":
        return TraditionalDecoder(self)

    def mutate(self, rate: float = 0.4) -> "TraditionalDecoderArch":
        out = copy.deepcopy(self)
        for i in range(4):
            if random.random() < rate:
                out.refine_ops[i] = random.choice(REFINE_CHOICES)
        for i in range(3):
            if random.random() < rate:
                out.fusion_ops[i] = random.choice(FUSION_CHOICES)
            if random.random() < rate:
                out.upsample_ops[i] = random.choice(UPSAMPLE_CHOICES)
        return out

    def crossover(self, other: "TraditionalDecoderArch") -> Tuple["TraditionalDecoderArch", "TraditionalDecoderArch"]:
        a = copy.deepcopy(self)
        b = copy.deepcopy(other)
        cut_ref = random.randint(1, 3)
        a.refine_ops[cut_ref:], b.refine_ops[cut_ref:] = b.refine_ops[cut_ref:], a.refine_ops[cut_ref:]
        cut_mid = random.randint(1, 2)
        a.fusion_ops[cut_mid:], b.fusion_ops[cut_mid:] = b.fusion_ops[cut_mid:], a.fusion_ops[cut_mid:]
        a.upsample_ops[cut_mid:], b.upsample_ops[cut_mid:] = b.upsample_ops[cut_mid:], a.upsample_ops[cut_mid:]
        return a, b

    def flops(self, input_h: int = 512, input_w: int = 512) -> int:
        channels = self.channels
        total = 0
        for op, scale in zip(self.refine_ops, STAGE_SCALES):
            h, w = max(1, input_h // scale), max(1, input_w // scale)
            if op == "sepconv3":
                total += h * w * (9 * channels + channels * channels)
            elif op == "mbconv":
                total += h * w * (channels * channels * 8 + 9 * channels * 4)
            elif op == "resblock":
                total += h * w * 18 * channels * channels
            else:
                total += h * w * 9 * channels * channels
        for mode, scale in zip(self.fusion_ops, (16, 8, 4)):
            h, w = max(1, input_h // scale), max(1, input_w // scale)
            if mode in {"concat", "gated"}:
                total += h * w * 2 * channels * channels
        h2, w2 = max(1, input_h // 2), max(1, input_w // 2)
        total += 9 * h2 * w2 * channels * max(channels // 2, 1)
        return int(total)

    def params(self) -> int:
        return sum(p.numel() for p in self.materialize().parameters())

    def n_nodes(self) -> int:
        return len(self.refine_ops) + len(self.fusion_ops) + len(self.upsample_ops)

    def summary(self) -> str:
        ref = ",".join(self.refine_ops)
        fus = ",".join(self.fusion_ops)
        up = ",".join(self.upsample_ops)
        return f"traditional_fpn(ref=[{ref}],fuse=[{fus}],up=[{up}])"

    def to_dict(self) -> dict:
        return {
            "kind": "traditional_fpn",
            "channels": self.channels,
            "refine_ops": list(self.refine_ops),
            "fusion_ops": list(self.fusion_ops),
            "upsample_ops": list(self.upsample_ops),
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }

    @staticmethod
    def from_summary(s: str, channels: int = 256) -> "TraditionalDecoderArch":
        """Parse the string produced by summary(), e.g.
        traditional_fpn(ref=[mbconv,sepconv3],fuse=[concat,gated],up=[deconv,bilinear])
        """
        import re
        ref = re.search(r"ref=\[([^\]]*)\]", s)
        fuse = re.search(r"fuse=\[([^\]]*)\]", s)
        up = re.search(r"up=\[([^\]]*)\]", s)
        if not (ref and fuse and up):
            raise ValueError(f"Cannot parse traditional_fpn summary: {s!r}")
        return TraditionalDecoderArch(
            channels=channels,
            refine_ops=[x.strip() for x in ref.group(1).split(",") if x.strip()],
            fusion_ops=[x.strip() for x in fuse.group(1).split(",") if x.strip()],
            upsample_ops=[x.strip() for x in up.group(1).split(",") if x.strip()],
        )

    @staticmethod
    def from_dict(d: dict) -> "TraditionalDecoderArch":
        if d.get("kind") != "traditional_fpn":
            raise ValueError(f"Unsupported traditional arch kind: {d.get('kind')}")
        return TraditionalDecoderArch(
            channels=d["channels"],
            refine_ops=list(d["refine_ops"]),
            fusion_ops=list(d["fusion_ops"]),
            upsample_ops=list(d["upsample_ops"]),
            min_depth=float(d.get("min_depth", 1e-3)),
            max_depth=float(d.get("max_depth", 80.0)),
        )


class TraditionalDecoder(nn.Module):
    def __init__(self, arch: TraditionalDecoderArch):
        super().__init__()
        c = arch.channels
        self.refines = nn.ModuleList([make_refine(name, c) for name in arch.refine_ops])
        self.fusions = nn.ModuleList([Fusion(name, c) for name in arch.fusion_ops])
        self.ups = nn.ModuleList([Up(name, c) for name in arch.upsample_ops])
        self.up_to_half = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.head = _AdaBinsHead(c, min_depth=arch.min_depth, max_depth=arch.max_depth)

    @staticmethod
    def _target_size(token_hw: Tuple[int, int], scale: int) -> Tuple[int, int]:
        h_tok, w_tok = token_hw
        return (
            max(1, int(round(h_tok * PATCH_SIZE / scale))),
            max(1, int(round(w_tok * PATCH_SIZE / scale))),
        )

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, encoder_feats: List[Tensor]) -> Dict[str, Tensor]:
        z0, z1, z2, z3 = encoder_feats[:4]
        token_hw = z0.shape[-2:]
        p4 = self._resize(z0, self._target_size(token_hw, 4))
        p8 = self._resize(z1, self._target_size(token_hw, 8))
        p16 = self._resize(z2, self._target_size(token_hw, 16))
        p32 = self._resize(z3, self._target_size(token_hw, 32))

        x32 = self.refines[0](p32)
        x16 = self.refines[1](self.fusions[0](self.ups[0](x32, p16.shape[-2:]), p16))
        x8 = self.refines[2](self.fusions[1](self.ups[1](x16, p8.shape[-2:]), p8))
        x4 = self.refines[3](self.fusions[2](self.ups[2](x8, p4.shape[-2:]), p4))
        x2 = self.up_to_half(F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False))
        depth = self.head(x2)
        return {
            "scale_1_32": x32,
            "scale_1_16": x16,
            "scale_1_8": x8,
            "scale_1_4": x4,
            "scale_1_2": x2,
            "e0": x4,
            "depth": depth,
        }
