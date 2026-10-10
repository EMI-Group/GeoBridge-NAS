"""Tree-coded Mamba cells inside a fixed DINO decoder trunk.

The search space is deliberately narrow:
- the decoder macro skeleton is fixed;
- upsampling, lateral fusion, and depth head are fixed plumbing;
- NAS only searches four stage-local tree cells made of Mamba operators.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components import SpatialMambaLayer


STAGE_SCALES = (32, 16, 8, 4)
PATCH_SIZE = 14

PATH_CHOICES_COARSE = ("fwd_bwd", "fwd_bwd_hw")
PATH_CHOICES_FINE = ("fwd", "fwd_bwd")
FUSE_CHOICES = ("scalar", "input_gate")
SHARE_CHOICES = ("independent", "share_x")
STATE_FUSION_CHOICES = ("none", "post")
D_CONV_CHOICES = (3, 5)
D_STATE_CHOICES = (8, 16)
DEPTH_CHOICES = (1, 2)


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
    def random(stage_idx: int) -> "MambaOpConfig":
        coarse = stage_idx <= 1
        return MambaOpConfig(
            depth=random.choice(DEPTH_CHOICES if coarse else (1,)),
            d_state=random.choice(D_STATE_CHOICES),
            path_set=random.choice(PATH_CHOICES_COARSE if coarse else PATH_CHOICES_FINE),
            fuse_mode=random.choice(FUSE_CHOICES),
            share_mode=random.choice(SHARE_CHOICES),
            state_fusion_pos=random.choice(STATE_FUSION_CHOICES),
            d_conv=random.choice(D_CONV_CHOICES),
        )

    def mutate(self, stage_idx: int) -> None:
        field_name = random.choice([
            "depth", "d_state", "path_set", "fuse_mode",
            "share_mode", "state_fusion_pos", "d_conv",
        ])
        coarse = stage_idx <= 1
        if field_name == "depth":
            self.depth = random.choice(DEPTH_CHOICES if coarse else (1,))
        elif field_name == "d_state":
            self.d_state = random.choice(D_STATE_CHOICES)
        elif field_name == "path_set":
            self.path_set = random.choice(PATH_CHOICES_COARSE if coarse else PATH_CHOICES_FINE)
        elif field_name == "fuse_mode":
            self.fuse_mode = random.choice(FUSE_CHOICES)
        elif field_name == "share_mode":
            self.share_mode = random.choice(SHARE_CHOICES)
        elif field_name == "state_fusion_pos":
            self.state_fusion_pos = random.choice(STATE_FUSION_CHOICES)
        elif field_name == "d_conv":
            self.d_conv = random.choice(D_CONV_CHOICES)

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "d_state": self.d_state,
            "path_set": self.path_set,
            "fuse_mode": self.fuse_mode,
            "share_mode": self.share_mode,
            "state_fusion_pos": self.state_fusion_pos,
            "prop_bias": self.prop_bias,
            "d_conv": self.d_conv,
        }

    @staticmethod
    def from_dict(d: dict) -> "MambaOpConfig":
        return MambaOpConfig(**d)

    def validate(self, stage_idx: int) -> bool:
        coarse = stage_idx <= 1
        if self.depth not in (DEPTH_CHOICES if coarse else (1,)):
            return False
        if self.d_state not in D_STATE_CHOICES:
            return False
        if self.path_set not in (PATH_CHOICES_COARSE if coarse else PATH_CHOICES_FINE):
            return False
        return (
            self.fuse_mode in FUSE_CHOICES
            and self.share_mode in SHARE_CHOICES
            and self.state_fusion_pos in STATE_FUSION_CHOICES
            and self.d_conv in D_CONV_CHOICES
        )

    def short(self) -> str:
        path = {"fwd": "F", "fwd_bwd": "FB", "fwd_bwd_hw": "HW"}[self.path_set]
        fuse = {"scalar": "S", "input_gate": "G"}[self.fuse_mode]
        share = {"independent": "I", "share_x": "X"}[self.share_mode]
        pos = {"none": "N", "post": "O"}[self.state_fusion_pos]
        return f"M{self.depth}D{self.d_state}{path}{fuse}{share}{pos}K{self.d_conv}"

    @staticmethod
    def from_short(text: str) -> "MambaOpConfig":
        if not text.startswith("M"):
            raise ValueError(f"Invalid Mamba op code: {text}")
        d_pos = text.find("D", 1)
        k_pos = text.rfind("K")
        if d_pos < 0 or k_pos < 0:
            raise ValueError(f"Invalid Mamba op code: {text}")

        depth = int(text[1:d_pos])
        body = text[d_pos + 1:k_pos]
        d_conv = int(text[k_pos + 1:])

        path_prefix = None
        for prefix in ("HW", "FB", "F"):
            split = body.find(prefix)
            if split > 0:
                path_prefix = prefix
                d_state = int(body[:split])
                flags = body[split + len(prefix):]
                break
        if path_prefix is None or len(flags) != 3:
            raise ValueError(f"Invalid Mamba op code: {text}")

        path_set = {"F": "fwd", "FB": "fwd_bwd", "HW": "fwd_bwd_hw"}[path_prefix]
        fuse_mode = {"S": "scalar", "G": "input_gate"}[flags[0]]
        share_mode = {"I": "independent", "X": "share_x"}[flags[1]]
        state_fusion_pos = {"N": "none", "O": "post"}[flags[2]]
        return MambaOpConfig(
            depth=depth,
            d_state=d_state,
            path_set=path_set,
            fuse_mode=fuse_mode,
            share_mode=share_mode,
            state_fusion_pos=state_fusion_pos,
            d_conv=d_conv,
        )


@dataclass
class MambaCellNode:
    kind: str
    op: Optional[MambaOpConfig] = None
    children: List["MambaCellNode"] = field(default_factory=list)

    @staticmethod
    def random(stage_idx: int, max_depth: int = 2) -> "MambaCellNode":
        if max_depth <= 0:
            return MambaCellNode("op", op=MambaOpConfig.random(stage_idx))
        kind = random.choices(
            ["op", "res", "seq", "par"],
            weights=[0.45, 0.25, 0.20, 0.10],
            k=1,
        )[0]
        if kind == "op":
            return MambaCellNode("op", op=MambaOpConfig.random(stage_idx))
        if kind == "res":
            return MambaCellNode("res", children=[MambaCellNode.random(stage_idx, max_depth - 1)])
        return MambaCellNode(
            kind,
            children=[
                MambaCellNode.random(stage_idx, max_depth - 1),
                MambaCellNode.random(stage_idx, max_depth - 1),
            ],
        )

    def mutate(self, stage_idx: int, max_depth: int = 2) -> "MambaCellNode":
        node = copy.deepcopy(self)
        action = random.choice(["op_field", "replace", "child"])
        if action == "replace":
            return MambaCellNode.random(stage_idx, max_depth)
        if action == "op_field" and node.kind == "op" and node.op is not None:
            node.op.mutate(stage_idx)
            return node
        if node.children:
            child_idx = random.randrange(len(node.children))
            node.children[child_idx] = node.children[child_idx].mutate(stage_idx, max_depth - 1)
            return node
        if node.op is not None:
            node.op.mutate(stage_idx)
        return node

    def validate(self, stage_idx: int) -> bool:
        if self.kind == "op":
            return self.op is not None and self.op.validate(stage_idx) and not self.children
        if self.kind == "res":
            return len(self.children) == 1 and self.children[0].validate(stage_idx)
        if self.kind in {"seq", "par"}:
            return len(self.children) == 2 and all(c.validate(stage_idx) for c in self.children)
        return False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "op": self.op.to_dict() if self.op is not None else None,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: dict) -> "MambaCellNode":
        return MambaCellNode(
            kind=d["kind"],
            op=MambaOpConfig.from_dict(d["op"]) if d.get("op") is not None else None,
            children=[MambaCellNode.from_dict(c) for c in d.get("children", [])],
        )

    def summary(self) -> str:
        if self.kind == "op":
            return self.op.short() if self.op is not None else "EMPTY"
        if self.kind == "res":
            return f"R({self.children[0].summary()})"
        sep = ">" if self.kind == "seq" else "+"
        return "(" + sep.join(c.summary() for c in self.children) + ")"

    def count_ops(self) -> int:
        if self.kind == "op":
            return 1
        return sum(c.count_ops() for c in self.children)

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
                self.pos += 1
            elif self._peek("+"):
                kind = "par"
                self.pos += 1
            else:
                raise ValueError(f"Expected '>' or '+' at position {self.pos} in {self.text!r}")
            right = self.parse_node()
            self._expect(")")
            return MambaCellNode(kind, children=[left, right])

        return MambaCellNode("op", op=MambaOpConfig.from_short(self._read_op()))

    def expect_end(self) -> None:
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.pos} in {self.text!r}")

    def _read_op(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ")>+":
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"Expected Mamba op at position {self.pos} in {self.text!r}")
        return self.text[start:self.pos]

    def _peek(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def _expect(self, token: str) -> None:
        if not self._peek(token):
            raise ValueError(f"Expected {token!r} at position {self.pos} in {self.text!r}")
        self.pos += len(token)


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
            assert node.op is not None
            self.op = MambaOpModule(channels, node.op)
        else:
            self.children_mod = nn.ModuleList([MambaCellModule(c, channels) for c in node.children])

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
            return sum(child(x) for child in self.children_mod) / len(self.children_mod)
        raise RuntimeError(f"Unknown Mamba cell node kind: {self.kind}")


class _AdaBinsHead(nn.Module):
    """AdaBins-style adaptive depth-bin head (Bhat et al., CVPR 2021).

    Two parallel paths share the decoder feature map:

      * Global path (mViT)  →  per-image bin widths (B, N)
        Patch-embed + AvgPool downsamples the dense map to a small token grid,
        a tiny transformer encoder reasons over it, mean-pool yields a global
        token, and a small MLP outputs N positive widths via softmax. Widths
        cumsum into bin edges spanning [min_depth, max_depth]; centers are
        midpoints.

      * Local path  →  per-pixel softmax over bins (B, N, H, W)
        A 3×3 → 1×1 conv stack outputs per-pixel logits, softmaxed across the
        bin dimension.

    Final depth = Σ_i P_i(x, y) · center_i(image), strictly bounded in
    [min_depth, max_depth] and adapted per image.

    Why this is supposed to lift d1 over the sigmoid head: hard pixels (far
    sky / windows / poles) usually live in a narrow part of the depth range
    that the adaptive bin layout can resolve more finely than a uniform
    sigmoid, and per-image bin centers let the model express the bimodal
    near/far distribution typical of outdoor scenes.
    """

    def __init__(
        self,
        channels: int,
        min_depth: float,
        max_depth: float,
        n_bins: int = 256,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        downsample_stride: int = 16,
        min_bin_width: float = 1e-3,
    ):
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.n_bins = int(n_bins)
        self.embed_dim = int(embed_dim)
        self.min_bin_width = float(min_bin_width)

        # mViT: cheap patch-embed (1×1 + avg pool to keep token count small),
        # then transformer encoder.
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=downsample_stride, stride=downsample_stride),
        )
        # Learnable positional encoding for up to 4096 tokens (auto-interpolated
        # if exceeded — covers 352×1216 ÷ (2·16) = 11×38 = 418 tokens safely).
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transformer_norm = nn.LayerNorm(embed_dim)

        self.bin_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_bins),
        )

        # Local path: per-pixel bin logits. Bottleneck to channels//2 first to
        # keep the cost on the order of the previous _BoundedDepthHead while
        # outputting N=n_bins channels.
        hidden = max(channels // 2, 1)
        self.pixel_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, n_bins, kernel_size=1),
        )

    def _pos_embed_for(self, L: int) -> Tensor:
        if L <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :L, :]
        # Auto-interpolate if input is larger than the cached length.
        return F.interpolate(
            self.pos_embed.transpose(1, 2),
            size=L,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, _, _, _ = x.shape

        # --- Global path: bin widths ---
        z = self.patch_embed(x)                       # (B, E, H', W')
        z = z.flatten(2).transpose(1, 2)              # (B, L, E)
        z = z + self._pos_embed_for(z.shape[1])
        z = self.transformer_norm(self.transformer(z))
        g = z.mean(dim=1)                              # (B, E)

        raw = F.softmax(self.bin_head(g), dim=-1)     # (B, N), sums to 1
        total = self.max_depth - self.min_depth
        reserved = self.min_bin_width * self.n_bins
        widths = raw * (total - reserved) + self.min_bin_width   # ≥ min_bin_width

        edges_inc = torch.cumsum(widths, dim=-1)      # (B, N)
        edges = torch.cat(
            [torch.zeros(B, 1, dtype=widths.dtype, device=widths.device), edges_inc],
            dim=-1,
        ) + self.min_depth                             # (B, N+1) in [min, max]
        centers = 0.5 * (edges[:, :-1] + edges[:, 1:])  # (B, N)

        # --- Local path: per-pixel softmax over bins ---
        logits = self.pixel_conv(x)                   # (B, N, H, W)
        probs = F.softmax(logits, dim=1)              # softmax over bin axis

        depth = (probs * centers.view(B, self.n_bins, 1, 1)).sum(
            dim=1, keepdim=True
        )                                              # (B, 1, H, W)
        return depth


class FixedMambaDecoderTrunk(nn.Module):
    """DINO pyramid adapter + fixed top-down Mamba decoder trunk."""

    def __init__(
        self,
        stage_cells: Optional[List[MambaCellNode]] = None,
        channels: int = 256,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        stage_modules: Optional[List[nn.Module]] = None,
    ):
        super().__init__()
        if stage_modules is None:
            assert stage_cells is not None and len(stage_cells) == 4
        else:
            assert len(stage_modules) == 4
        self.channels = channels
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            for _ in range(4)
        ])
        self.cells = nn.ModuleList(
            [MambaCellModule(cell, channels) for cell in stage_cells]
            if stage_modules is None else stage_modules
        )
        self.fuse16 = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse8 = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse4 = nn.Conv2d(channels, channels, 1, bias=False)
        self.up_to_half = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.head = _AdaBinsHead(channels, min_depth=min_depth, max_depth=max_depth)

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
        assert len(encoder_feats) >= 4, "FixedMambaDecoderTrunk needs four DINO layer features"
        z0, z1, z2, z3 = encoder_feats[:4]
        token_hw = z0.shape[-2:]

        p4 = self.adapters[0](self._resize(z0, self._target_size(token_hw, 4)))
        p8 = self.adapters[1](self._resize(z1, self._target_size(token_hw, 8)))
        p16 = self.adapters[2](self._resize(z2, self._target_size(token_hw, 16)))
        p32 = self.adapters[3](self._resize(z3, self._target_size(token_hw, 32)))

        x32 = self.cells[0](p32)
        x16 = self.cells[1](self.fuse16(self._resize(x32, p16.shape[-2:]) + p16))
        x8 = self.cells[2](self.fuse8(self._resize(x16, p8.shape[-2:]) + p8))
        x4 = self.cells[3](self.fuse4(self._resize(x8, p4.shape[-2:]) + p4))

        x2 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x2 = self.up_to_half(x2)
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


class MaterializedDecoder(nn.Module):
    def __init__(
        self,
        stage_cells: List[MambaCellNode],
        channels: int = 256,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
    ):
        super().__init__()
        self.trunk = FixedMambaDecoderTrunk(
            stage_cells, channels, min_depth=min_depth, max_depth=max_depth
        )

    def forward(self, encoder_feats: List[Tensor]) -> Dict[str, Tensor]:
        return self.trunk(encoder_feats)


class DecoderTree:
    """Architecture genotype for fixed-trunk, tree-coded Mamba cell search."""

    def __init__(
        self,
        channels: int = 256,
        stage_cells: Optional[List[MambaCellNode]] = None,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
    ):
        self.channels = channels
        # Physical depth bounds for the sigmoid-bounded head. NOT to be confused
        # with the per-cell tree-recursion `max_depth` argument elsewhere.
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.stage_cells = stage_cells or [
            MambaCellNode.random(i, max_depth=1 if i >= 2 else 2)
            for i in range(4)
        ]

    def validate(self) -> bool:
        return len(self.stage_cells) == 4 and all(
            cell.validate(i) for i, cell in enumerate(self.stage_cells)
        )

    def materialize(self) -> MaterializedDecoder:
        return MaterializedDecoder(
            copy.deepcopy(self.stage_cells),
            self.channels,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    @staticmethod
    def _stage_hw(input_h: int, input_w: int, scale: int) -> Tuple[int, int]:
        return max(1, input_h // scale), max(1, input_w // scale)

    def _cell_complexity(self, node: MambaCellNode, h: int, w: int) -> int:
        channels = self.channels
        if node.kind == "op" and node.op is not None:
            paths = {"fwd": 1, "fwd_bwd": 2, "fwd_bwd_hw": 4}[node.op.path_set]
            per_block = h * w * channels * channels * (8 + paths)
            per_block += h * w * channels * node.op.d_conv * node.op.d_conv
            return int(per_block * node.op.depth)
        return sum(self._cell_complexity(ch, h, w) for ch in node.children)

    def flops(self, input_h: int = 512, input_w: int = 512) -> int:
        channels = self.channels
        total = 0
        for stage_idx, scale in enumerate(STAGE_SCALES):
            h, w = self._stage_hw(input_h, input_w, scale)
            total += 9 * h * w * channels * channels
            total += self._cell_complexity(self.stage_cells[stage_idx], h, w)
        h2, w2 = self._stage_hw(input_h, input_w, 2)
        total += 9 * h2 * w2 * channels * channels
        total += 9 * h2 * w2 * channels * max(channels // 2, 1)
        return int(total)

    def params(self) -> int:
        return sum(p.numel() for p in self.materialize().parameters())

    def n_nodes(self) -> int:
        return sum(cell.count_ops() for cell in self.stage_cells)

    def summary(self) -> str:
        return "|".join(
            f"s{scale}:{cell.summary()}"
            for scale, cell in zip(STAGE_SCALES, self.stage_cells)
        )

    @staticmethod
    def from_summary(text: str, channels: int = 256) -> "DecoderTree":
        parts = [part.strip() for part in text.strip().split("|")]
        if len(parts) != len(STAGE_SCALES):
            raise ValueError(f"Expected {len(STAGE_SCALES)} stage specs, got {len(parts)}: {text}")

        by_scale: Dict[int, MambaCellNode] = {}
        for part in parts:
            if ":" not in part or not part.startswith("s"):
                raise ValueError(f"Invalid stage spec: {part}")
            stage_name, expr = (x.strip() for x in part.split(":", 1))
            scale = int(stage_name[1:])
            if scale not in STAGE_SCALES:
                raise ValueError(f"Unsupported stage scale s{scale}; expected {STAGE_SCALES}")
            by_scale[scale] = MambaCellNode.from_summary(expr)

        missing = [scale for scale in STAGE_SCALES if scale not in by_scale]
        if missing:
            raise ValueError(f"Missing stage specs for scales: {missing}")

        arch = DecoderTree(channels=channels, stage_cells=[by_scale[scale] for scale in STAGE_SCALES])
        if not arch.validate():
            raise ValueError(f"Invalid genotype for this search space: {text}")
        return arch

    def to_dict(self) -> dict:
        return {
            "kind": "tree_coded_mamba_trunk",
            "channels": self.channels,
            "stage_cells": [c.to_dict() for c in self.stage_cells],
        }

    @staticmethod
    def from_dict(d: dict) -> "DecoderTree":
        if d.get("kind") != "tree_coded_mamba_trunk":
            raise ValueError(f"Unsupported architecture kind: {d.get('kind')}")
        return DecoderTree(
            channels=d["channels"],
            stage_cells=[MambaCellNode.from_dict(c) for c in d["stage_cells"]],
        )

    def __repr__(self) -> str:
        return f"DecoderTree(kind=tree_coded_mamba_trunk, stages={self.summary()})"


def make_fpn_template(channels: int = 256) -> DecoderTree:
    """Compatibility name for callers; returns the default fixed Mamba trunk."""
    cells = [
        MambaCellNode("res", children=[MambaCellNode("op", op=MambaOpConfig.random(0))]),
        MambaCellNode("res", children=[MambaCellNode("op", op=MambaOpConfig.random(1))]),
        MambaCellNode("op", op=MambaOpConfig.random(2)),
        MambaCellNode("op", op=MambaOpConfig.random(3)),
    ]
    return DecoderTree(channels=channels, stage_cells=cells)


def make_unet_template(channels: int = 256) -> DecoderTree:
    return make_fpn_template(channels)


class TreeOperator:
    """Stage-wise genetic operators for tree-coded Mamba cell genotypes."""

    @staticmethod
    def crossover(t1: DecoderTree, t2: DecoderTree) -> Tuple[DecoderTree, DecoderTree]:
        c1 = copy.deepcopy(t1)
        c2 = copy.deepcopy(t2)
        stage = random.randrange(4)
        c1.stage_cells[stage], c2.stage_cells[stage] = c2.stage_cells[stage], c1.stage_cells[stage]
        return c1, c2

    @staticmethod
    def mutate(t: DecoderTree, mutation_rate: float = 0.3) -> DecoderTree:
        t_new = copy.deepcopy(t)
        for i in range(4):
            if random.random() < mutation_rate:
                max_depth = 1 if i >= 2 else 2
                t_new.stage_cells[i] = t_new.stage_cells[i].mutate(i, max_depth=max_depth)
        return t_new if t_new.validate() else t

    @staticmethod
    def random_tree(
        channels: int = 256,
        template_fn: Callable = make_fpn_template,
    ) -> DecoderTree:
        return DecoderTree(
            channels=channels,
            stage_cells=[
                MambaCellNode.random(i, max_depth=1 if i >= 2 else 2)
                for i in range(4)
            ],
        )


def random_valid_tree(channels: int = 256) -> DecoderTree:
    return TreeOperator.random_tree(channels)
