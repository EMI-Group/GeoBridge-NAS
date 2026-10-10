"""Internal Discretization modules from iDisc (Piccinelli et al., ICCV 2023).

Vendored from third_party/idisc/idisc/models/id_module.py, with einops calls
rewritten in pure torch and `build` classmethods removed (the head wrapper
in head.py constructs these directly).

Two modules:

- AFP (Adaptive Foundation Projection): learnable scene tokens that get
  refined by iterative slot-attention over the pixel features of each
  resolution. Returns one set of (B, num_slots, latent_dim) per resolution.

- ISD (Internal Slot Decoder): per-resolution head that does iterative
  cross-attention from pixel features → slot tokens, then projects to
  log-depth (output_dim=1).
"""

from typing import Tuple

import torch
import torch.nn as nn

from ._attention import AttentionLayer
from ._layers import _get_activation_cls, get_norm
from ._pos_embed import PositionEmbeddingSine


def _bchw_to_bnc(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    return x.flatten(2).transpose(1, 2).contiguous()


def _bnc_to_bchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)."""
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w).contiguous()


class ISDHead(nn.Module):
    """Per-resolution head: pixel queries → slot keys/values → log-depth."""

    def __init__(
        self,
        depth: int,
        pixel_dim: int = 256,
        query_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pixel_pe = PositionEmbeddingSine(pixel_dim // 2, normalize=True)

        for i in range(self.depth):
            setattr(
                self, f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=pixel_dim, hidden_dim=pixel_dim,
                    source_dim=query_dim, output_dim=pixel_dim,
                    num_heads=num_heads, dropout=0.0,
                    pre_norm=True, sink_competition=False,
                ),
            )
            setattr(
                self, f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )

        self.proj_output = nn.Sequential(
            get_norm(norm, pixel_dim),
            nn.Linear(pixel_dim, pixel_dim),
            get_norm(norm, pixel_dim),
            nn.Linear(pixel_dim, output_dim),
        )

    def forward(self, feature_map: torch.Tensor, idrs: torch.Tensor) -> torch.Tensor:
        _, _, h, w = feature_map.shape
        x = feature_map + self.pixel_pe(feature_map)
        x = _bchw_to_bnc(x)

        for i in range(self.depth):
            update = getattr(self, f"cross_attn_{i+1}")(x.clone(), idrs)
            x = x + update
            x = x + getattr(self, f"mlp_{i+1}")(x.clone())

        out = self.proj_output(x)              # (B, H*W, output_dim)
        return _bnc_to_bchw(out, h, w)         # (B, output_dim, H, W)


class ISD(nn.Module):
    """Multi-resolution wrapper: one ISDHead per pyramid level."""

    def __init__(
        self,
        num_resolutions: int,
        depth: int,
        pixel_dim: int = 128,
        query_dim: int = 128,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        for i in range(num_resolutions):
            setattr(
                self, f"head_{i+1}",
                ISDHead(
                    depth=depth, pixel_dim=pixel_dim, query_dim=query_dim,
                    num_heads=num_heads, output_dim=output_dim,
                    expansion=expansion, activation=activation, norm=norm,
                ),
            )

    def forward(
        self, xs: Tuple[torch.Tensor, ...], idrs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, f"head_{i+1}")(xs[i], idrs[i])
            for i in range(self.num_resolutions)
        )


class AFP(nn.Module):
    """Adaptive Foundation Projection: learnable slot tokens refined by
    iterative slot-attention over pixel features (one set of slots per scale).
    """

    def __init__(
        self,
        num_resolutions: int,
        depth: int = 3,
        pixel_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps

        bottleneck_dim = expansion * latent_dim

        for i in range(self.num_resolutions):
            setattr(
                self, f"pixel_pe_{i+1}",
                PositionEmbeddingSine(pixel_dim // 2, normalize=True),
            )
            setattr(
                self, f"mu_{i+1}",
                nn.Parameter(torch.randn(1, self.num_slots, latent_dim)),
            )
            # Upstream re-uses the same cross-attn module across iters
            # (the `_d1` suffix is a leftover from an earlier variant).
            setattr(
                self, f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=latent_dim, hidden_dim=latent_dim,
                    source_dim=pixel_dim, output_dim=latent_dim,
                    num_heads=num_heads, dropout=0.0,
                    pre_norm=True, sink_competition=True,
                ),
            )
            setattr(
                self, f"mlp_cross_{i+1}",
                nn.Sequential(
                    get_norm(norm, latent_dim),
                    nn.Linear(latent_dim, bottleneck_dim),
                    _get_activation_cls(activation),
                    nn.Linear(bottleneck_dim, latent_dim),
                ),
            )

    def forward(
        self, feature_maps: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        b = feature_maps[0].shape[0]
        idrs = []
        feature_maps_flat = []

        for i in range(self.num_resolutions):
            fmap = feature_maps[i]
            fmap_pe = fmap + getattr(self, f"pixel_pe_{i+1}")(fmap)
            feature_maps_flat.append(_bchw_to_bnc(fmap_pe))
            idrs.append(getattr(self, f"mu_{i+1}").expand(b, -1, -1))

        for i in range(self.num_resolutions):
            cross_attn = getattr(self, f"cross_attn_{i+1}")
            mlp = getattr(self, f"mlp_cross_{i+1}")
            for _ in range(self.iters):
                idrs[i] = idrs[i] + cross_attn(idrs[i].clone(), feature_maps_flat[i])
                idrs[i] = idrs[i] + mlp(idrs[i].clone())

        return tuple(idrs)
