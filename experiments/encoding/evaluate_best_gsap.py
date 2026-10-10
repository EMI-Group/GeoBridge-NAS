#!/usr/bin/env python3
"""Fair post-search GSAP re-evaluation for tree versus full-vector encoding."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import flat_full_vector  # noqa: F401
from ddn.search.fitness_eval import build_evaluator_from_cache


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score_front(path: Path, cache_dir: str, grad_batch_size: int, seed: int) -> list[dict]:
    with path.open("rb") as handle:
        front = pickle.load(handle)
    rows = []
    for index, arch in enumerate(front):
        seed_all(seed)
        evaluator = build_evaluator_from_cache(cache_dir, list(range(512)), "cuda", 256, grad_batch_size)
        score = evaluator(arch, generation=-1)
        rows.append({"arch_idx": index, "arch": arch.summary(), "gsp_raw": float(score.score_grad), "gsp_objective": float(score.score_grad_stability), "flops": float(score.flops), "params": float(score.params)})
        del evaluator
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_pareto", required=True)
    parser.add_argument("--flat_pareto", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grad_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tree = score_front(Path(args.tree_pareto), args.cache_dir, args.grad_batch_size, args.seed)
    flat = score_front(Path(args.flat_pareto), args.cache_dir, args.grad_batch_size, args.seed)
    payload = {"protocol": {"criterion": "minimum GSAP objective = log10(gradient sensitivity)", "shared_cache_samples": 512, "grad_batch_size": args.grad_batch_size, "seed_reset_per_candidate": args.seed, "note": "Both encodings are post-search re-evaluated with the identical evaluator."}, "tree": {"best": min(tree, key=lambda item: item["gsp_objective"]), "all_candidates": tree}, "flat": {"best": min(flat, key=lambda item: item["gsp_objective"]), "all_candidates": flat}}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"tree_best": payload["tree"]["best"], "flat_best": payload["flat"]["best"]}))


if __name__ == "__main__":
    main()
