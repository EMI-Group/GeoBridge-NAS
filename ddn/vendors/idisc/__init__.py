"""Vendored iDisc head (Piccinelli et al., ICCV 2023).

Only the head modules are vendored — pixel encoder, pixel decoder, losses and
data loaders from the upstream repo are not needed here. The original code uses
einops; this vendor copy rewrites all `rearrange` calls in plain torch to avoid
a runtime dependency.

Upstream: https://github.com/SysCV/idisc (CC-BY-NC 4.0 / ECL-2.0).
"""

from .head import IDiscHead

__all__ = ["IDiscHead"]
