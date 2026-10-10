"""iDisc head: AFP slot tokens + per-scale ISD log-depth + multi-scale average.

Input:  list/tuple of multi-scale feature maps, **ordered coarse→fine**
        (e.g. [/16, /8, /4]). Every map must have channel dim == pixel_dim.
Output: (B, 1, H_top, W_top) depth in [min_depth, max_depth], where (H_top,
        W_top) is the spatial shape of the finest input scale. The caller is
        responsible for any further upsampling to the network input size.

Defaults below mirror the published iDisc-ViT-S depth configuration:
  num_slots=32, afp_iters=3, isd_depth=2, num_heads=4, expansion=4.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._id_module import AFP, ISD


class IDiscHead(nn.Module):
    def __init__(
        self,
        pixel_dim: int = 256,
        num_resolutions: int = 3,
        num_slots: int = 32,
        latent_dim: int = 256,
        afp_iters: int = 3,
        isd_depth: int = 2,
        num_heads: int = 4,
        expansion: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
    ):
        super().__init__()
        self.num_resolutions = int(num_resolutions)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        self.afp = AFP(
            num_resolutions=self.num_resolutions,
            depth=afp_iters,
            pixel_dim=pixel_dim,
            latent_dim=latent_dim,
            num_latents=num_slots,
            num_heads=num_heads,
            activation=activation,
            norm=norm,
            expansion=expansion,
        )
        self.isd = ISD(
            num_resolutions=self.num_resolutions,
            depth=isd_depth,
            pixel_dim=pixel_dim,
            query_dim=latent_dim,
            num_heads=num_heads,
            output_dim=1,
            expansion=expansion,
            activation=activation,
            norm=norm,
        )

    def forward(self, feature_maps: Sequence[Tensor]) -> Tensor:
        if len(feature_maps) != self.num_resolutions:
            raise ValueError(
                f"IDiscHead expects {self.num_resolutions} feature maps, "
                f"got {len(feature_maps)}"
            )

        idrs = self.afp(tuple(feature_maps))
        outs: List[Tensor] = list(self.isd(tuple(feature_maps), idrs))
        # ISD returns log-depth per pixel; upstream takes exp() then averages
        # all scales after upsampling to the finest one.
        top_h, top_w = outs[-1].shape[-2:]
        depths = [
            F.interpolate(
                torch.exp(o), size=(top_h, top_w),
                mode="bilinear", align_corners=True,
            )
            for o in outs
        ]
        depth = torch.stack(depths, dim=0).mean(dim=0)
        return depth.clamp(self.min_depth, self.max_depth)
