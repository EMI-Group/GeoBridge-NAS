"""Fixed DDN subnet structures adapted to COS3D instance-field training.

The transferred object is the searched decoder structure, not its depth-task
weights.  The DINOv2 ViT-S encoder stays frozen, while the selected A1/A2/A3
decoder is re-initialized and optimized on COS3D.  Its dense 16-D prediction
acts as a view-conditioned training branch for the rendered Gaussian instance
field.  COS3D inference still uses the learned per-Gaussian 16-D field, so the
transfer branch introduces no inference-time dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components import SpatialMambaLayer
from .vendor.dinov2 import DINOv2


PATCH_SIZE = 14
FEATURE_DIM = 384
INTERMEDIATE_LAYERS = [2, 5, 8, 11]
STAGE_SCALES = (32, 16, 8, 4)

# Exact structures selected by the existing DDN search.  Stage order is
# coarse-to-fine: s32 -> s16 -> s8 -> s4.
ARCHITECTURE_CODES: Dict[str, str] = {
    "a1": (
        "s32:M1D16FBGXNK5|s16:M1D16FBSINK3|"
        "s8:M1D8FGIOK3|s4:R(M1D16FGINK3)"
    ),
    "a2": (
        "s32:M1D16FBGINK5|s16:M1D16FBSIOK3|"
        "s8:M1D8FGIOK3|s4:M1D8FBSIOK3"
    ),
    "a3": (
        "s32:((M2D8HWGXOK5>M1D16HWGXOK3)>R(M2D8FBSIOK3))|"
        "s16:(M2D8FBGXNK3+R(M1D16HWGIOK3))|"
        "s8:R(M1D8FSXNK3)|s4:M1D16FGINK3"
    ),
}


@dataclass
class MambaOpConfig:
    depth: int = 1
    d_state: int = 16
    path_set: str = "fwd_bwd"
    fuse_mode: str = "scalar"
    share_mode: str = "share_x"
    state_fusion_pos: str = "post"
    prop_bias: float = 0.5
    d_conv: int = 3

    @staticmethod
    def from_short(text: str) -> "MambaOpConfig":
        if not text.startswith("M"):
            raise ValueError(f"Invalid Mamba op code: {text}")
        d_pos = text.find("D", 1)
        k_pos = text.rfind("K")
        if d_pos < 0 or k_pos < 0:
            raise ValueError(f"Invalid Mamba op code: {text}")

        depth = int(text[1:d_pos])
        body = text[d_pos + 1 : k_pos]
        d_conv = int(text[k_pos + 1 :])
        path_prefix: Optional[str] = None
        flags = ""
        for prefix in ("HW", "FB", "F"):
            split = body.find(prefix)
            if split > 0:
                path_prefix = prefix
                d_state = int(body[:split])
                flags = body[split + len(prefix) :]
                break
        if path_prefix is None or len(flags) != 3:
            raise ValueError(f"Invalid Mamba op code: {text}")

        return MambaOpConfig(
            depth=depth,
            d_state=d_state,
            path_set={"F": "fwd", "FB": "fwd_bwd", "HW": "fwd_bwd_hw"}[
                path_prefix
            ],
            fuse_mode={"S": "scalar", "G": "input_gate"}[flags[0]],
            share_mode={"I": "independent", "X": "share_x"}[flags[1]],
            state_fusion_pos={"N": "none", "O": "post"}[flags[2]],
            d_conv=d_conv,
        )


@dataclass
class MambaCellNode:
    kind: str
    op: Optional[MambaOpConfig] = None
    children: List["MambaCellNode"] = field(default_factory=list)

    @staticmethod
    def from_summary(text: str) -> "MambaCellNode":
        parser = _MambaCellSummaryParser(text)
        node = parser.parse_node()
        parser.expect_end()
        return node


class _MambaCellSummaryParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse_node(self) -> MambaCellNode:
        if self._peek("R("):
            self.pos += 2
            child = self.parse_node()
            self._expect(")")
            return MambaCellNode("res", children=[child])
        if self._peek("("):
            self.pos += 1
            left = self.parse_node()
            if self._peek(">"):
                kind = "seq"
            elif self._peek("+"):
                kind = "par"
            else:
                raise ValueError(
                    f"Expected '>' or '+' at position {self.pos} in {self.text!r}"
                )
            self.pos += 1
            right = self.parse_node()
            self._expect(")")
            return MambaCellNode(kind, children=[left, right])
        return MambaCellNode("op", op=MambaOpConfig.from_short(self._read_op()))

    def expect_end(self) -> None:
        if self.pos != len(self.text):
            raise ValueError(
                f"Unexpected trailing text at position {self.pos} in {self.text!r}"
            )

    def _read_op(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ")>+":
            self.pos += 1
        if start == self.pos:
            raise ValueError(
                f"Expected Mamba op at position {self.pos} in {self.text!r}"
            )
        return self.text[start : self.pos]

    def _peek(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def _expect(self, token: str) -> None:
        if not self._peek(token):
            raise ValueError(
                f"Expected {token!r} at position {self.pos} in {self.text!r}"
            )
        self.pos += len(token)


def parse_architecture(code: str) -> List[MambaCellNode]:
    by_scale: Dict[int, MambaCellNode] = {}
    for part in code.split("|"):
        scale_text, expr = part.split(":", 1)
        scale = int(scale_text[1:])
        by_scale[scale] = MambaCellNode.from_summary(expr)
    missing = [scale for scale in STAGE_SCALES if scale not in by_scale]
    if missing:
        raise ValueError(f"Architecture is missing stages: {missing}")
    return [by_scale[scale] for scale in STAGE_SCALES]


class MambaOpModule(nn.Module):
    def __init__(self, channels: int, cfg: MambaOpConfig):
        super().__init__()
        self.layer = SpatialMambaLayer(
            dim=channels,
            depth=cfg.depth,
            d_state=cfg.d_state,
            mlp_ratio=4.0,
            ssm_kwargs={
                "path_set": cfg.path_set,
                "fuse_mode": cfg.fuse_mode,
                "share_mode": cfg.share_mode,
                "state_fusion_pos": cfg.state_fusion_pos,
                "prop_bias": cfg.prop_bias,
                "d_conv": cfg.d_conv,
            },
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class MambaCellModule(nn.Module):
    def __init__(self, node: MambaCellNode, channels: int):
        super().__init__()
        self.kind = node.kind
        if node.kind == "op":
            if node.op is None:
                raise ValueError("Mamba op node has no configuration")
            self.op = MambaOpModule(channels, node.op)
        else:
            self.children_mod = nn.ModuleList(
                [MambaCellModule(child, channels) for child in node.children]
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "op":
            return self.op(x)
        if self.kind == "res":
            return x + self.children_mod[0](x)
        if self.kind == "seq":
            for child in self.children_mod:
                x = child(x)
            return x
        if self.kind == "par":
            return sum(child(x) for child in self.children_mod) / len(
                self.children_mod
            )
        raise RuntimeError(f"Unknown Mamba cell node kind: {self.kind}")


class FixedInstanceDecoder(nn.Module):
    """The DDN top-down macro trunk with a COS3D 16-D embedding head."""

    def __init__(
        self,
        architecture: str,
        channels: int = 256,
        output_dim: int = 16,
    ):
        super().__init__()
        if architecture not in ARCHITECTURE_CODES:
            raise ValueError(
                f"Unknown architecture {architecture!r}; "
                f"choose from {sorted(ARCHITECTURE_CODES)}"
            )
        cells = parse_architecture(ARCHITECTURE_CODES[architecture])
        self.architecture = architecture
        self.channels = channels
        self.scale_proj = nn.ModuleList(
            [nn.Conv2d(FEATURE_DIM, channels, 1, bias=False) for _ in range(4)]
        )
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.cells = nn.ModuleList(
            [MambaCellModule(cell, channels) for cell in cells]
        )
        self.fuse16 = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse8 = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse4 = nn.Conv2d(channels, channels, 1, bias=False)
        self.up_to_half = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.instance_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, output_dim, 1),
        )

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

    def forward(
        self,
        encoder_feats: List[Tensor],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if len(encoder_feats) < 4:
            raise ValueError("FixedInstanceDecoder needs four DINOv2 features")
        projected = [
            proj(feat)
            for proj, feat in zip(self.scale_proj, encoder_feats[:4])
        ]
        z0, z1, z2, z3 = projected
        token_hw = z0.shape[-2:]
        p4 = self.adapters[0](
            self._resize(z0, self._target_size(token_hw, 4))
        )
        p8 = self.adapters[1](
            self._resize(z1, self._target_size(token_hw, 8))
        )
        p16 = self.adapters[2](
            self._resize(z2, self._target_size(token_hw, 16))
        )
        p32 = self.adapters[3](
            self._resize(z3, self._target_size(token_hw, 32))
        )

        x32 = self.cells[0](p32)
        x16 = self.cells[1](
            self.fuse16(self._resize(x32, p16.shape[-2:]) + p16)
        )
        x8 = self.cells[2](
            self.fuse8(self._resize(x16, p8.shape[-2:]) + p8)
        )
        x4 = self.cells[3](
            self.fuse4(self._resize(x8, p4.shape[-2:]) + p4)
        )
        x2 = self.up_to_half(
            F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        )
        embedding = self.instance_head(x2)
        if output_size is not None and embedding.shape[-2:] != output_size:
            embedding = F.interpolate(
                embedding,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return F.normalize(embedding, dim=1, eps=1e-6)


class FrozenDINOv2(nn.Module):
    """Frozen ViT-S feature extractor matching the existing DDN encoder."""

    def __init__(self, weight_path: str):
        super().__init__()
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {path}")
        self.model = DINOv2("vits")
        state = torch.load(path, map_location="cpu")
        for wrapper in ("model", "state_dict", "teacher", "student"):
            if (
                isinstance(state, dict)
                and wrapper in state
                and isinstance(state[wrapper], dict)
            ):
                state = state[wrapper]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        matched = len(self.model.state_dict()) - len(missing)
        if matched == 0:
            raise RuntimeError(
                f"No DINOv2 ViT-S weights matched checkpoint {path}"
            )
        if unexpected:
            print(
                f"[DDN transfer] ignored {len(unexpected)} unexpected DINO keys"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, image: Tensor) -> List[Tensor]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = (image - self.image_mean) / self.image_std
        height, width = image.shape[-2:]
        pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        patch_h, patch_w = image.shape[-2] // PATCH_SIZE, image.shape[-1] // PATCH_SIZE
        raw = self.model.get_intermediate_layers(
            image,
            INTERMEDIATE_LAYERS,
            return_class_token=True,
        )
        features: List[Tensor] = []
        for patch_tokens, _ in raw:
            feature = patch_tokens.reshape(
                image.shape[0], patch_h, patch_w, FEATURE_DIM
            )
            features.append(feature.permute(0, 3, 1, 2).contiguous())
        return features


class DDNTransferBranch(nn.Module):
    """Frozen DINOv2 plus a target-task-reinitialized A1/A2/A3 decoder."""

    def __init__(
        self,
        architecture: str,
        dino_weight_path: str,
        channels: int = 256,
        output_dim: int = 16,
    ):
        super().__init__()
        self.architecture = architecture
        self.encoder = FrozenDINOv2(dino_weight_path)
        self.decoder = FixedInstanceDecoder(
            architecture=architecture,
            channels=channels,
            output_dim=output_dim,
        )
        # DINOv2 is frozen, and every COS3D camera is revisited many times.
        # Cache the four maps in host FP16 memory after their first use.
        self._feature_cache: Dict[str, List[Tensor]] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        image: Tensor,
        output_size: Tuple[int, int],
        cache_key: Optional[str] = None,
    ) -> Tensor:
        if cache_key is not None and cache_key in self._feature_cache:
            features = [
                feature.to(device=image.device, dtype=torch.float32)
                for feature in self._feature_cache[cache_key]
            ]
        else:
            features = self.encoder(image)
            if cache_key is not None:
                self._feature_cache[cache_key] = [
                    feature.detach().to(device="cpu", dtype=torch.float16)
                    for feature in features
                ]
        return self.decoder(features, output_size=output_size)

    def trainable_parameters(self):
        return self.decoder.parameters()

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.decoder.parameters()
            if parameter.requires_grad
        )
