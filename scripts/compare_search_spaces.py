#!/usr/bin/env python3
"""Search-space-level ablation under the same proxy and NSGA-II budget.

Spaces:
  mamba_tree       Current proposed OP/RES/SEQ/PAR Mamba tree space.
  traditional_fpn  Conventional flat FPN/U-Net decoder NAS space with CNN ops.
"""

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.proxy.grad_stability import ArchScore, compute_fitness, normalize_objectives
from ddn.search.fitness_eval import build_evaluator_from_cache
from ddn.search_space.traditional import TraditionalDecoderArch
from ddn.search_space.tree import DecoderTree, TreeOperator, random_valid_tree


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
Objectives = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--space", choices=["mamba_tree", "traditional_fpn"], required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--proxy", default="gsp")
    p.add_argument("--pop_size", type=int, default=64)
    p.add_argument("--n_generations", type=int, default=30)
    p.add_argument("--n_eval_samples", type=int, default=512)
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument("--crossover_rate", type=float, default=0.7)
    p.add_argument("--mutation_rate", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--forward_batch_size", type=int, default=4)
    p.add_argument("--grad_batch_size", type=int, default=8)
    p.add_argument("--dinov2_ckpt", default=None)
    return p.parse_args()


def make_random(space: str, channels: int):
    if space == "mamba_tree":
        return random_valid_tree(channels)
    if space == "traditional_fpn":
        return TraditionalDecoderArch.random(channels)
    raise ValueError(space)


def make_offspring(space: str, p1, p2, crossover_rate: float, mutation_rate: float):
    if space == "mamba_tree":
        if random.random() < crossover_rate:
            c1, c2 = TreeOperator.crossover(p1, p2)
        else:
            c1, c2 = p1, p2
        return (
            TreeOperator.mutate(c1, mutation_rate),
            TreeOperator.mutate(c2, mutation_rate),
        )
    if space == "traditional_fpn":
        if random.random() < crossover_rate:
            c1, c2 = p1.crossover(p2)
        else:
            c1, c2 = p1, p2
        return c1.mutate(mutation_rate), c2.mutate(mutation_rate)
    raise ValueError(space)


def dominates(a: Objectives, b: Objectives) -> bool:
    better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            better = True
    return better


def non_dominated_sort(objectives: Sequence[Objectives]) -> List[List[int]]:
    n = len(objectives)
    dominated = [[] for _ in range(n)]
    count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding(fronts: List[List[int]], objectives: Sequence[Objectives]) -> List[float]:
    out = [0.0] * len(objectives)
    if not objectives:
        return out
    n_obj = len(objectives[0])
    for front in fronts:
        if len(front) <= 2:
            for idx in front:
                out[idx] = float("inf")
            continue
        for m in range(n_obj):
            order = sorted(front, key=lambda i: objectives[i][m])
            out[order[0]] = float("inf")
            out[order[-1]] = float("inf")
            span = objectives[order[-1]][m] - objectives[order[0]][m]
            if span == 0:
                continue
            for k in range(1, len(order) - 1):
                out[order[k]] += (objectives[order[k + 1]][m] - objectives[order[k - 1]][m]) / span
    return out


def tournament(population: list, fronts: List[List[int]], crowd: List[float]):
    rank = {}
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r
    i, j = random.sample(range(len(population)), 2)
    return population[min((i, j), key=lambda x: (rank.get(x, len(fronts)), -crowd[x]))]


def environmental_selection(population: list, objectives: Sequence[Objectives], size: int) -> list:
    fronts = non_dominated_sort(objectives)
    crowd = crowding(fronts, objectives)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            selected.extend(sorted(front, key=lambda i: -crowd[i])[: size - len(selected)])
            break
    return [population[i] for i in selected]


def log_generation(gen: int, population: list, scores: List[ArchScore], fronts: List[List[int]], log_rows: list):
    pareto = set(fronts[0]) if fronts else set()
    individuals = []
    for i, (arch, score) in enumerate(zip(population, scores)):
        individuals.append({
            "idx": i,
            "pareto": i in pareto,
            "grad": score.score_grad,
            "grad_obj": score.score_grad_stability,
            "flops": score.flops,
            "flops_g": score.flops / 1e9,
            "params": score.params,
            "params_m": score.params / 1e6,
            "arch": arch.summary(),
        })
    avg_obj = sum(x.score_grad_stability for x in scores) / max(len(scores), 1)
    LOGGER.info("Gen %03d | Pareto=%d/%d | avg_obj=%.4f", gen, len(pareto), len(scores), avg_obj)
    log_rows.append({
        "gen": gen,
        "n_pareto": len(pareto),
        "avg_grad_obj": avg_obj,
        "individuals": individuals,
    })


def evaluate_population(population: list, evaluator: Callable, gen: int) -> Tuple[List[ArchScore], List[Objectives]]:
    scores = [evaluator(arch, gen) for arch in population]
    for i, score in enumerate(scores):
        score.arch_id = i
    return scores, normalize_objectives(compute_fitness(scores, gen))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_eval_samples < args.grad_batch_size:
        args.n_eval_samples = args.grad_batch_size
    evaluator = build_evaluator_from_cache(
        cache_dir=args.cache_dir,
        batch_indices=list(range(args.n_eval_samples)),
        device=args.device,
        decoder_channels=args.decoder_channels,
        forward_batch_size=args.forward_batch_size,
        grad_batch_size=args.grad_batch_size,
        dinov2_weight_path=args.dinov2_ckpt,
        proxy=args.proxy,
        proxy_seed=args.seed,
    )

    population = [make_random(args.space, args.decoder_channels) for _ in range(args.pop_size)]
    log_rows = []
    for gen in range(args.n_generations):
        scores, objectives = evaluate_population(population, evaluator, gen)
        fronts = non_dominated_sort(objectives)
        crowd = crowding(fronts, objectives)
        log_generation(gen, population, scores, fronts, log_rows)
        (out_dir / "search_log.json").write_text(json.dumps(log_rows, indent=2))

        offspring = []
        while len(offspring) < args.pop_size:
            p1 = tournament(population, fronts, crowd)
            p2 = tournament(population, fronts, crowd)
            c1, c2 = make_offspring(args.space, p1, p2, args.crossover_rate, args.mutation_rate)
            offspring.extend([c1, c2])
        offspring = offspring[: args.pop_size]

        off_scores, _ = evaluate_population(offspring, evaluator, gen)
        combined = population + offspring
        combined_obj = normalize_objectives(
            compute_fitness(scores, gen) + compute_fitness(off_scores, gen)
        )
        population = environmental_selection(combined, combined_obj, args.pop_size)

    final_scores, final_obj = evaluate_population(population, evaluator, -1)
    final_fronts = non_dominated_sort(final_obj)
    pareto_indices = final_fronts[0] if final_fronts else list(range(len(population)))
    pareto_front = [population[i] for i in pareto_indices]

    with open(out_dir / "pareto_front.pkl", "wb") as f:
        pickle.dump(pareto_front, f)
    summary = [
        {
            "arch_idx": i,
            "flops": float(arch.flops(480, 640)),
            "params": float(arch.params()),
            "n_nodes": arch.n_nodes(),
            "arch": arch.summary(),
        }
        for i, arch in enumerate(pareto_front)
    ]
    (out_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps(vars(args), indent=2))
    LOGGER.info("Done. Pareto front size=%d -> %s", len(pareto_front), out_dir)


if __name__ == "__main__":
    main()
