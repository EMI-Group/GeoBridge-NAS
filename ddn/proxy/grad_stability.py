"""Gradient-stability proxy objectives for NAS."""

import math
from dataclasses import dataclass
from typing import List, Tuple

MIN_VALID_GRAD = 1e-12


def grad_stability_objective(grad: float) -> float:
    """Lower is better, except near-zero dead gradients are penalized."""
    grad = float(grad)
    if grad <= MIN_VALID_GRAD:
        return 30.0
    return math.log10(grad)


@dataclass
class ArchScore:
    """Stores per-architecture proxy scores and hardware costs."""

    arch_id: int
    score_grad: float = 0.0
    score_grad_stability: float = 0.0
    flops: float = 0.0
    params: float = 0.0


def compute_fitness(
    population: List[ArchScore],
    generation: int,
) -> List[Tuple[float, float, float]]:
    """
    Compute NSGA-II objectives for each architecture (all minimize, raw scale).

    Returns list of (grad_stability_obj, flops, params) tuples.
    grad_stability_obj = log10(eps + gradient sensitivity).  Smaller is better:
    in this search space, large early gradients indicate over-sensitive and
    less stable decoders rather than better trainability.
    Call normalize_objectives() on combined populations before Pareto sorting.
    """
    objectives = []
    for arch in population:
        grad_obj = grad_stability_objective(arch.score_grad)
        arch.score_grad_stability = grad_obj
        objectives.append((grad_obj, arch.flops, arch.params))

    return objectives


def normalize_objectives(
    objectives: List[Tuple[float, ...]],
) -> List[Tuple[float, ...]]:
    """
    Min-max normalize each objective dimension to [0, 1].

    Ensures FLOPs, Params and gradient-stability objective are compared on the
    same scale before Pareto dominance.  Call this on the combined
    parent+offspring set so both are referenced to the same range.
    """
    if len(objectives) < 2:
        return objectives
    n_obj = len(objectives[0])
    mins = [min(o[j] for o in objectives) for j in range(n_obj)]
    maxs = [max(o[j] for o in objectives) for j in range(n_obj)]
    result = []
    for o in objectives:
        normed = tuple(
            (o[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] > mins[j] else 0.0
            for j in range(n_obj)
        )
        result.append(normed)
    return result
