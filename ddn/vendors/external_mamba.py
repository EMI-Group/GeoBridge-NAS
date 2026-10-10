"""Adapters for official VMamba and LocalMamba blocks in the DDN decoder.

The DDN pyramid, lateral fusion, and AdaBins head remain unchanged.  This file
only adapts the public operator implementations to the BCHW decoder interface.
It deliberately exposes no architecture-search or tuning knobs.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Type

import torch.nn as nn

from ..search_space.tree import FixedMambaDecoderTrunk


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    """Load an official module under an isolated package name."""
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _official_block(kind: str) -> Type[nn.Module]:
    if kind == "vmamba":
        path = _ROOT / "third_party" / "VMamba" / "classification" / "models" / "vmamba.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official VMamba source: {path}")
        return _load_module("_ddn_official_vmamba", path).VSSBlock
    if kind == "localmamba":
        path = _ROOT / "third_party" / "LocalMamba" / "classification" / "lib" / "models" / "local_vmamba.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official LocalMamba source: {path}")
        return _load_module("_ddn_official_localmamba", path).VSSBlock
    raise ValueError(f"Unsupported external Mamba baseline: {kind}")


class OfficialMambaCell(nn.Module):
    """One unchanged official block, with only BCHW/BHWC layout adaptation."""

    # LocalMamba's released no-search LocalViM preset.  Passing no directions
    # would activate its directional-search supernet, which is not a baseline.
    _LOCALMAMBA_NOSEARCH_DIRECTIONS = ("h", "h_flip", "w2", "w2_flip")

    def __init__(self, channels: int, kind: str):
        super().__init__()
        block = _official_block(kind)
        if kind == "vmamba":
            # Published VMamba-v2 Tiny configuration (the official release
            # default, rather than the generic VSSBlock constructor defaults).
            self.block = block(
                hidden_dim=channels,
                ssm_d_state=1,
                ssm_ratio=2.0,
                ssm_conv=3,
                ssm_conv_bias=False,
                forward_type="v05_noz",
                mlp_ratio=4.0,
            )
        else:
            # Official LocalMamba VSSBlock defaults, plus the published fixed
            # no-search direction preset above (window sizes 2 and 2).
            self.block = block(hidden_dim=channels, directions=self._LOCALMAMBA_NOSEARCH_DIRECTIONS)

    def forward(self, x):
        return self.block(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ExternalMambaDecoder(nn.Module):
    """Fixed DDN decoder scaffold populated with one official block per scale."""

    def __init__(self, kind: str, channels: int, min_depth: float, max_depth: float):
        super().__init__()
        if kind not in {"vmamba", "localmamba"}:
            raise ValueError(f"Unsupported external Mamba baseline: {kind}")
        self.kind = kind
        self.trunk = FixedMambaDecoderTrunk(
            channels=channels,
            min_depth=min_depth,
            max_depth=max_depth,
            stage_modules=[OfficialMambaCell(channels, kind) for _ in range(4)],
        )

    def forward(self, encoder_feats) -> Dict[str, object]:
        return self.trunk(encoder_feats)
