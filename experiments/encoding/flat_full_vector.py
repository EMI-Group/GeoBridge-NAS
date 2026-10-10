"""Conventional fixed-length, repair-free vector encoding of the bounded tree grammar.

All seven heap slots per stage remain part of the genotype, including slots that
are inactive under the decoded topology.  Crossover and mutation operate on
these independent vector loci and decoding never rewrites them.  This provides
the ordinary-vector baseline for the tree's topology-aware operators.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ddn.search_space.tree import DecoderTree, MambaCellNode, MambaOpConfig

SLOTS = 7
KINDS = ("op", "res", "seq", "par")


def max_depth(stage: int) -> int:
    return 1 if stage >= 2 else 2


def random_token(stage: int) -> Dict[str, Any]:
    return {"kind": random.choices(KINDS, weights=(0.45, 0.25, 0.20, 0.10), k=1)[0], "op": MambaOpConfig.random(stage).to_dict()}


def decode(slots: List[Dict[str, Any]], stage: int, idx: int = 0, depth: int | None = None) -> MambaCellNode:
    if depth is None:
        depth = max_depth(stage)
    token = slots[idx]
    kind = token["kind"] if depth > 0 else "op"
    if kind == "op":
        return MambaCellNode("op", op=MambaOpConfig.from_dict(token["op"]))
    left = decode(slots, stage, idx * 2 + 1, depth - 1)
    if kind == "res":
        return MambaCellNode("res", children=[left])
    right = decode(slots, stage, idx * 2 + 2, depth - 1)
    return MambaCellNode(kind, children=[left, right])


@dataclass
class FlatFullVectorArch:
    stage_slots: List[List[Dict[str, Any]]]
    channels: int = 256
    min_depth: float = 1e-3
    max_depth: float = 80.0

    def __post_init__(self) -> None:
        if len(self.stage_slots) != 4 or any(len(stage) != SLOTS for stage in self.stage_slots):
            raise ValueError("FlatFullVectorArch requires 4 x 7 fixed vector slots")

    @classmethod
    def random(cls, channels: int = 256) -> "FlatFullVectorArch":
        return cls([[random_token(stage) for _ in range(SLOTS)] for stage in range(4)], channels)

    def to_native(self) -> DecoderTree:
        return DecoderTree(channels=self.channels, stage_cells=[decode(stage, i) for i, stage in enumerate(self.stage_slots)], min_depth=self.min_depth, max_depth=self.max_depth)

    def validate(self) -> bool:
        return self.to_native().validate()

    def materialize(self):
        return self.to_native().materialize()

    def flops(self, *args, **kwargs) -> int:
        return self.to_native().flops(*args, **kwargs)

    def params(self) -> int:
        return self.to_native().params()

    def n_nodes(self) -> int:
        return self.to_native().n_nodes()

    def summary(self) -> str:
        return self.to_native().summary()

    def to_dict(self) -> dict:
        return {"kind": "flat_full_vector_bounded_tree", "channels": self.channels, "min_depth": self.min_depth, "max_depth": self.max_depth, "stage_slots": copy.deepcopy(self.stage_slots)}

    @classmethod
    def from_dict(cls, data: dict) -> "FlatFullVectorArch":
        if data.get("kind") != "flat_full_vector_bounded_tree":
            raise ValueError(f"Unexpected flat encoding kind: {data.get('kind')}")
        return cls(data["stage_slots"], data["channels"], data.get("min_depth", 1e-3), data.get("max_depth", 80.0))


class FlatFullVectorOperator:
    """One-point vector crossover and independent vector-locus mutation."""

    @staticmethod
    def crossover(a: FlatFullVectorArch, b: FlatFullVectorArch) -> Tuple[FlatFullVectorArch, FlatFullVectorArch]:
        left, right = copy.deepcopy(a.stage_slots), copy.deepcopy(b.stage_slots)
        cut = random.randrange(1, 4 * SLOTS)
        for linear in range(cut, 4 * SLOTS):
            stage, slot = divmod(linear, SLOTS)
            left[stage][slot], right[stage][slot] = right[stage][slot], left[stage][slot]
        return FlatFullVectorArch(left, a.channels, a.min_depth, a.max_depth), FlatFullVectorArch(right, b.channels, b.min_depth, b.max_depth)

    @staticmethod
    def mutate(arch: FlatFullVectorArch, mutation_rate: float = 0.3) -> FlatFullVectorArch:
        out = copy.deepcopy(arch)
        for stage in range(4):
            if random.random() >= mutation_rate:
                continue
            token = out.stage_slots[stage][random.randrange(SLOTS)]
            if random.random() < 1.0 / 8.0:
                token["kind"] = random.choice(KINDS)
            else:
                config = MambaOpConfig.from_dict(token["op"])
                config.mutate(stage)
                token["op"] = config.to_dict()
        return out


def random_valid_flat(channels: int = 256) -> FlatFullVectorArch:
    return FlatFullVectorArch.random(channels)
