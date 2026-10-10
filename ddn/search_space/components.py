"""Mamba operator primitives for the searched decoder cells.

This module intentionally contains only the operators used by the NAS search
space.  Plumbing such as upsampling, skip fusion, and depth heads is fixed in
``tree.py`` and is not part of the candidate library.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_PATH_SETS = {
    "fwd": ("fwd",),
    "fwd_bwd": ("fwd", "bwd"),
    "fwd_bwd_hw": ("fwd", "bwd", "h_fwd", "h_bwd"),
}


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class StateFusion(nn.Module):
    """Local state mixer used by the StructureAwareSSM fallback."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class StructureAwareSSM(nn.Module):
    """Configurable multi-path spatial SSM-style mixer.

    The search-facing knobs mirror the intended Mamba decoder design.  The
    implementation uses a deterministic cumulative-scan fallback so the proxy
    evaluator can run without a custom CUDA selective-scan backend.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        path_set: str = "fwd_bwd",
        fuse_mode: str = "scalar",
        share_mode: str = "share_x",
        state_fusion_pos: str = "post",
        prop_bias: float = 0.5,
        max_ssm_tokens: int = 4096,
    ):
        super().__init__()
        if path_set not in _PATH_SETS:
            raise ValueError(f"path_set={path_set} not in {list(_PATH_SETS)}")
        if fuse_mode not in {"scalar", "input_gate"}:
            raise ValueError(f"fuse_mode={fuse_mode} invalid")
        if share_mode not in {"independent", "share_x"}:
            raise ValueError(f"share_mode={share_mode} invalid")
        if state_fusion_pos not in {"post", "none"}:
            raise ValueError(f"state_fusion_pos={state_fusion_pos} invalid")

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.path_names = list(_PATH_SETS[path_set])
        self.n_paths = len(self.path_names)
        self.fuse_mode = fuse_mode
        self.share_mode = share_mode
        self.state_fusion_pos = state_fusion_pos
        self.prop_bias = float(prop_bias)
        self.max_ssm_tokens = max_ssm_tokens

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        if share_mode == "share_x":
            self.shared_x_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        for p_name in self.path_names:
            if share_mode != "share_x":
                setattr(self, f"x_proj_{p_name}", nn.Linear(self.d_inner, self.d_inner, bias=False))
            self.register_parameter(f"decay_{p_name}", nn.Parameter(torch.full((self.d_inner,), 0.5)))

        if fuse_mode == "scalar":
            init = torch.zeros(self.n_paths)
            if self.n_paths == 2:
                init[1] = -1.5
            self.scan_logits = nn.Parameter(init)
        else:
            self.gate_proj = nn.Linear(self.d_inner, self.n_paths)

        self.state_fusion = StateFusion(self.d_inner) if state_fusion_pos == "post" else None
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _get_x_proj(self, p_name: str) -> nn.Linear:
        if self.share_mode == "share_x":
            return self.shared_x_proj
        return getattr(self, f"x_proj_{p_name}")

    def _get_decay(self, p_name: str) -> Tensor:
        return getattr(self, f"decay_{p_name}")

    @staticmethod
    def _path_seq_to_canonical(t_seq: Tensor, path: str, height: int, width: int) -> Tensor:
        batch, channels, _ = t_seq.shape
        if path == "fwd":
            return t_seq
        if path == "bwd":
            return t_seq.flip(dims=[-1])
        if path == "h_fwd":
            return t_seq.reshape(batch, channels, width, height).transpose(2, 3).reshape(batch, channels, height * width)
        if path == "h_bwd":
            return t_seq.flip(dims=[-1]).reshape(batch, channels, width, height).transpose(2, 3).reshape(batch, channels, height * width)
        raise ValueError(path)

    @staticmethod
    def _xs_for_path(xs_canonical: Tensor, path: str, height: int, width: int) -> Tensor:
        batch, channels, _ = xs_canonical.shape
        if path == "fwd":
            return xs_canonical
        if path == "bwd":
            return xs_canonical.flip(dims=[-1])
        if path == "h_fwd":
            return xs_canonical.reshape(batch, channels, height, width).transpose(2, 3).reshape(batch, channels, width * height)
        if path == "h_bwd":
            return xs_canonical.reshape(batch, channels, height, width).transpose(2, 3).reshape(batch, channels, width * height).flip(dims=[-1])
        raise ValueError(path)

    def _path_weights(self, x_bchw: Tensor):
        if self.fuse_mode == "scalar":
            return torch.softmax(self.scan_logits, dim=0)
        pooled = x_bchw.mean(dim=(2, 3))
        return torch.softmax(self.gate_proj(pooled), dim=-1).unsqueeze(-1)

    def _scan_fallback(self, xs_p: Tensor, p_name: str) -> Tensor:
        x_proj = self._get_x_proj(p_name)
        x = x_proj(xs_p.transpose(1, 2)).transpose(1, 2)
        decay = torch.sigmoid(self._get_decay(p_name)).view(1, -1, 1)
        if p_name != "fwd":
            decay = (decay + self.prop_bias / 10.0).clamp(max=0.99)
        steps = torch.arange(xs_p.shape[-1], device=xs_p.device, dtype=xs_p.dtype).view(1, 1, -1)
        weights = decay.pow(steps)
        denom = torch.cumsum(weights.expand_as(x), dim=-1).clamp(min=1e-4)
        return torch.cumsum(x * weights, dim=-1) / denom

    def ssm(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        xs_canonical = x.reshape(batch, channels, height * width)
        path_outs = []
        for p_name in self.path_names:
            xs_p = self._xs_for_path(xs_canonical, p_name, height, width)
            y_p = self._scan_fallback(xs_p, p_name)
            y_p = self._path_seq_to_canonical(y_p, p_name, height, width)
            path_outs.append(y_p.reshape(batch, channels, height * width))

        weights = self._path_weights(x)
        if self.fuse_mode == "scalar":
            y = sum(weights[i] * path_outs[i] for i in range(self.n_paths))
        else:
            stacked = torch.stack(path_outs, dim=1)
            y = (weights.view(batch, self.n_paths, 1, 1) * stacked).sum(dim=1)

        if self.state_fusion is not None:
            y = self.state_fusion(y.reshape(batch, channels, height, width)).reshape(batch, channels, height * width)
        return y

    def _resize_for_ssm(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        _, _, height, width = x.shape
        if height * width <= self.max_ssm_tokens:
            return x, (height, width)
        scale = (self.max_ssm_tokens / float(height * width)) ** 0.5
        h_small = max(1, int(height * scale))
        w_small = max(1, int(width * scale))
        return F.interpolate(x, size=(h_small, w_small), mode="bilinear", align_corners=False), (height, width)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x_ssm, orig_hw = self._resize_for_ssm(x)
        h_small, w_small = x_ssm.shape[2], x_ssm.shape[3]
        y = self.ssm(x_ssm).reshape(batch, self.d_inner, h_small, w_small)
        if (h_small, w_small) != orig_hw:
            y = F.interpolate(y, size=orig_hw, mode="bilinear", align_corners=False)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        assert y.shape[-1] == self.d_inner
        return self.dropout(self.out_proj(y))


class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        ssm_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        ssm_kwargs = ssm_kwargs or {}
        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = StructureAwareSSM(
            d_model=hidden_dim,
            d_state=d_state,
            **ssm_kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.cpe1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class SpatialMambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        ssm_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
                hidden_dim=dim,
                d_state=d_state,
                mlp_ratio=mlp_ratio,
                ssm_kwargs=ssm_kwargs,
            )
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 3, 1, 2).contiguous()
