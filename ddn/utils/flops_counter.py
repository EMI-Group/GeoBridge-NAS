"""FLOPs and parameter counting utilities."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def count_flops_params(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 512, 512),
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Estimate FLOPs (GFLOPs) and parameter count (M) for a model.

    Uses a hook-based approach for conv and linear layers.

    Args:
        model: any nn.Module.
        input_shape: (B, C, H, W) input shape for the forward pass.

    Returns:
        (gflops, params_m) — GFLOPs and million parameters.
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    total_flops = [0]
    hooks = []

    def conv_hook(module: nn.Conv2d, inp, out):
        B, C_out, H_out, W_out = out.shape
        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        C_in = module.in_channels
        flops = B * C_out * H_out * W_out * (C_in // groups) * kH * kW * 2
        total_flops[0] += flops

    def linear_hook(module: nn.Linear, inp, out):
        B = inp[0].shape[0] if inp[0].dim() > 1 else 1
        flops = B * module.in_features * module.out_features * 2
        total_flops[0] += flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    dummy = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    gflops = total_flops[0] / 1e9
    return gflops, params_m
