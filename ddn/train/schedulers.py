"""Learning rate schedulers."""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    n_epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int = 1,
    eta_min_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing.

    Args:
        optimizer:        wrapped optimizer.
        n_epochs:         total training epochs.
        warmup_epochs:    number of epochs for linear warmup.
        steps_per_epoch:  if > 1, schedule is step-based; else epoch-based.
        eta_min_ratio:    final LR = base_lr * eta_min_ratio.
    """
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
