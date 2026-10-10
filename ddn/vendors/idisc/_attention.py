"""Multi-head attention with optional sink-competition (vendored from iDisc).

`sink_competition=True` switches the softmax axis so each value position
competes across the sink (slot) axis — this is what makes AFP behave like
slot attention.

Original used einops; rewritten in pure torch here.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim and num_heads are not divisible"
        self.eps = eps
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = (
            norm_layer(source_dim, eps=eps) if source_dim is not None else None
        )

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(
            hidden_dim, sink_dim if output_dim is None else output_dim
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition

    @staticmethod
    def _split_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # (B, N, H*D) -> (B*H, N, D)
        B, N, _ = t.shape
        return t.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).reshape(
            B * num_heads, N, head_dim
        )

    @staticmethod
    def _merge_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # (B*H, N, D) -> (B, N, H*D)
        BH, N, _ = t.shape
        B = BH // num_heads
        return t.reshape(B, num_heads, N, head_dim).permute(0, 2, 1, 3).reshape(
            B, N, num_heads * head_dim
        )

    def forward(
        self, sink: torch.Tensor, source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            sink = self.norm(sink)
            if source is not None:
                source = self.norm_context(source)

        q = self.to_q(sink)
        source = source if source is not None else sink
        k, v = self.to_kv(source).chunk(2, dim=-1)

        q = self._split_heads(q, self.num_heads, self.head_dim)
        k = self._split_heads(k, self.num_heads, self.head_dim)
        v = self._split_heads(v, self.num_heads, self.head_dim)

        sim = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if self.sink_competition:
            # softmax over the sink (slot) axis, then renormalize so each
            # value position's attention sums to 1 across slots.
            attn = F.softmax(sim, dim=-2) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
        else:
            attn = F.softmax(sim, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = self._merge_heads(out, self.num_heads, self.head_dim)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)
        return out
