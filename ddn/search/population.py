"""Population management utilities for NSGA-II."""

from typing import Dict, List, Optional, Tuple
import random

from ..search_space.tree import DecoderTree, TreeOperator, random_valid_tree
from ..proxy.grad_stability import ArchScore


class Population:
    """Manages a collection of (DecoderTree, ArchScore) pairs."""

    def __init__(self, size: int, channels: int = 256):
        self.size = size
        self.channels = channels
        self._archs: List[DecoderTree] = []
        self._scores: List[Optional[ArchScore]] = []

    def initialize(self) -> None:
        """Fill population with random valid trees."""
        self._archs = [random_valid_tree(self.channels) for _ in range(self.size)]
        self._scores = [None] * self.size

    def set_score(self, idx: int, score: ArchScore) -> None:
        self._scores[idx] = score

    def get_arch(self, idx: int) -> DecoderTree:
        return self._archs[idx]

    def get_score(self, idx: int) -> Optional[ArchScore]:
        return self._scores[idx]

    def best_by_grad_stability(self) -> Tuple[DecoderTree, ArchScore]:
        """Return the architecture with the lowest gradient-stability objective."""
        best_idx = min(
            range(len(self._archs)),
            key=lambda i: self._scores[i].score_grad_stability if self._scores[i] else float("inf"),
        )
        return self._archs[best_idx], self._scores[best_idx]

    def diversity_score(self) -> float:
        """Variance of gradient-stability objectives."""
        scores = [s.score_grad_stability for s in self._scores if s is not None]
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    def replace(self, new_archs: List[DecoderTree]) -> None:
        assert len(new_archs) == self.size
        self._archs = new_archs
        self._scores = [None] * self.size

    def __len__(self) -> int:
        return len(self._archs)

    def __iter__(self):
        return iter(self._archs)
