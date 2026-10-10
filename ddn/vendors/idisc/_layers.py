"""Layer helpers used by iDisc id_module (subset of upstream utils.layers)."""

from torch import nn


def _get_activation_cls(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    if activation == "silu":
        return nn.SiLU()
    raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}.")


def get_norm(norm, out_channels):
    if norm is None or (isinstance(norm, str) and len(norm) == 0):
        return None
    if isinstance(norm, str):
        factory = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "torchSyncBN": nn.SyncBatchNorm,
            "LN": lambda channels: nn.LayerNorm(channels),
            "torchLN": lambda channels: nn.LayerNorm(channels),
        }
        return factory[norm](out_channels)
    return norm(out_channels)
