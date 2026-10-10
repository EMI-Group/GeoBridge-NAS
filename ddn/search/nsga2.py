"""NSGA-II multi-objective optimizer for decoder architecture search."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch.distributed as dist

from ..search_space.tree import DecoderTree, TreeOperator, random_valid_tree
from ..proxy.grad_stability import ArchScore, compute_fitness, normalize_objectives

logger = logging.getLogger(__name__)

Objectives = Tuple[float, float, float]  # (grad_stability_obj, flops, params)


def _format_flops(flops: float) -> str:
    """Format FLOPs in G for readable search logs."""
    return f"{flops / 1e9:.3f}G"


def _format_params(params: float) -> str:
    """Format parameter count in M for readable search logs."""
    return f"{params / 1e6:.3f}M"


@dataclass
class NSGA2Config:
    pop_size: int = 64
    n_generations: int = 100
    crossover_rate: float = 0.7
    mutation_rate: float = 0.4
    tournament_size: int = 2
    decoder_channels: int = 256
    seed: int = 42
    log_every: int = 1
    log_path: Optional[Union[str, Path]] = None


class NSGA2:
    """
    NSGA-II multi-objective evolutionary search for decoder architectures.

    Three objectives (all minimize):
    1. Gradient stability objective: log10(grad sensitivity), smaller is better
    2. FLOPs
    3. Params
    """

    def __init__(self, config: NSGA2Config):
        self.config = config
        self.tree_op = TreeOperator()
        random.seed(config.seed)
        self._gen_log: List[Dict] = []
        self._log_path = Path(config.log_path) if config.log_path else None

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run_ddp(
        self,
        fitness_fn: Callable[[DecoderTree, int], ArchScore],
        rank: int,
        world_size: int,
    ) -> List[DecoderTree]:
        """
        DDP-parallel run: all ranks participate in fitness evaluation.

        Rank 0 drives the evolutionary logic; every rank evaluates a
        contiguous slice of the population / offspring each generation.
        DecoderTree and ArchScore are pickle-serialised for collective ops.

        Args:
            fitness_fn: callable(arch, generation) → ArchScore (runs locally).
            rank:       this process's DDP rank.
            world_size: total number of DDP ranks.
        """
        population: List[DecoderTree] = self._initialize_population() if rank == 0 else []

        for gen in range(self.config.n_generations):
            # ── Broadcast population (as dicts) so all ranks can scatter-eval ──
            pop_dicts = [[t.to_dict() for t in population]]
            dist.broadcast_object_list(pop_dicts, src=0)
            population = [DecoderTree.from_dict(d) for d in pop_dicts[0]]

            arch_scores = self._scatter_eval(population, gen, fitness_fn, rank, world_size)

            if rank == 0:
                for i, sc in enumerate(arch_scores):
                    sc.arch_id = i
                objectives_raw = compute_fitness(arch_scores, gen)
                objectives = normalize_objectives(objectives_raw)
                fronts = self._non_dominated_sort(objectives)
                crowd = self._crowding_distance_all(fronts, objectives)
                offspring_archs = self._make_offspring(population, fronts, crowd)
            else:
                offspring_archs = []
                objectives_raw = []

            # ── Broadcast offspring (as dicts) before evaluating ───────────────
            off_dicts = [[t.to_dict() for t in offspring_archs]]
            dist.broadcast_object_list(off_dicts, src=0)
            offspring_archs = [DecoderTree.from_dict(d) for d in off_dicts[0]]

            offspring_scores = self._scatter_eval(offspring_archs, gen, fitness_fn, rank, world_size)

            if rank == 0:
                offspring_objs_raw = compute_fitness(offspring_scores, gen)
                combined_pop = population + offspring_archs
                combined_obj = normalize_objectives(objectives_raw + offspring_objs_raw)
                logged_population = population
                population = self._environmental_selection(combined_pop, combined_obj)
                if gen % self.config.log_every == 0 or gen == self.config.n_generations - 1:
                    self._log_generation(gen, arch_scores, fronts, logged_population)
            else:
                population = []

        # ── Final evaluation ──────────────────────────────────────────────────
        pop_dicts = [[t.to_dict() for t in population]]
        dist.broadcast_object_list(pop_dicts, src=0)
        population = [DecoderTree.from_dict(d) for d in pop_dicts[0]]

        final_scores = self._scatter_eval(population, -1, fitness_fn, rank, world_size)

        if rank == 0:
            final_objs = normalize_objectives(compute_fitness(final_scores, -1))
            fronts = self._non_dominated_sort(final_objs)
            pareto_indices = fronts[0] if fronts else list(range(len(population)))
            return [population[i] for i in pareto_indices]

        dist.barrier()
        return []

    def _scatter_eval(
        self,
        archs: List[DecoderTree],
        gen: int,
        fitness_fn: Callable[[DecoderTree, int], ArchScore],
        rank: int,
        world_size: int,
    ) -> List[ArchScore]:
        """
        Collective: rank 0 scatters arch slices (as dicts), all ranks evaluate
        locally, rank 0 gathers and returns the full ordered score list.
        """
        if rank == 0:
            n = len(archs)
            per_rank = max(1, (n + world_size - 1) // world_size)
            arch_dicts = [t.to_dict() for t in archs]
            chunks: List[List[dict]] = []
            for r in range(world_size):
                start = r * per_rank
                chunks.append(arch_dicts[start : start + per_rank] if start < n else [])
        else:
            chunks = None  # type: ignore[assignment]

        my_chunk: List[Optional[List[dict]]] = [None]
        dist.scatter_object_list(my_chunk, chunks, src=0)
        my_archs: List[DecoderTree] = [
            DecoderTree.from_dict(d) for d in (my_chunk[0] or [])
        ]

        my_scores: List[ArchScore] = [fitness_fn(arch, gen) for arch in my_archs]

        gathered: Optional[List[List[ArchScore]]] = [None] * world_size if rank == 0 else None  # type: ignore[list-item]
        dist.gather_object(my_scores, gathered, dst=0)

        if rank == 0:
            return [s for chunk_scores in gathered for s in chunk_scores]  # type: ignore[union-attr]
        return []

    def run(
        self,
        fitness_fn: Callable[[DecoderTree, int], ArchScore],
    ) -> List[DecoderTree]:
        """
        Run NSGA-II search.

        Args:
            fitness_fn: callable(arch, generation) → ArchScore with
                        score_grad, flops, params filled.

        Returns:
            Pareto-front of DecoderTree architectures.
        """
        population = self._initialize_population()

        for gen in range(self.config.n_generations):
            # ── Evaluate ──
            arch_scores: List[ArchScore] = [
                fitness_fn(arch, gen) for arch in population
            ]
            for i, sc in enumerate(arch_scores):
                sc.arch_id = i

            objectives_raw = compute_fitness(arch_scores, gen)
            objectives = normalize_objectives(objectives_raw)

            # ── NSGA-II selection ──
            fronts = self._non_dominated_sort(objectives)
            crowd = self._crowding_distance_all(fronts, objectives)

            # ── Make offspring ──
            offspring_archs = self._make_offspring(population, fronts, crowd)
            offspring_scores: List[ArchScore] = [
                fitness_fn(arch, gen) for arch in offspring_archs
            ]
            offspring_objs_raw = compute_fitness(offspring_scores, gen)

            # ── Environmental selection (elitism) ──
            # Normalize combined so parent and offspring share the same scale reference.
            combined_pop = population + offspring_archs
            combined_obj = normalize_objectives(objectives_raw + offspring_objs_raw)
            logged_population = population
            population = self._environmental_selection(combined_pop, combined_obj)

            if gen % self.config.log_every == 0 or gen == self.config.n_generations - 1:
                self._log_generation(gen, arch_scores, fronts, logged_population)

        # Final evaluation
        final_scores: List[ArchScore] = [fitness_fn(a, -1) for a in population]
        final_objs = normalize_objectives(compute_fitness(final_scores, -1))
        fronts = self._non_dominated_sort(final_objs)

        pareto_indices = fronts[0] if fronts else list(range(len(population)))
        return [population[i] for i in pareto_indices]

    # ── Initialization ─────────────────────────────────────────────────────────

    def _initialize_population(self) -> List[DecoderTree]:
        pop = []
        for _ in range(self.config.pop_size):
            tree = random_valid_tree(self.config.decoder_channels)
            pop.append(tree)
        return pop

    # ── NSGA-II core ──────────────────────────────────────────────────────────

    def _non_dominated_sort(
        self, objectives: List[Objectives]
    ) -> List[List[int]]:
        """Return list of fronts, each front is a list of solution indices."""
        n = len(objectives)
        dominates = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    dominates[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                fronts[0].append(i)

        current = 0
        while fronts[current]:
            next_front = []
            for i in fronts[current]:
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    @staticmethod
    def _dominates(a: Objectives, b: Objectives) -> bool:
        """Return True if a Pareto-dominates b (all objectives ≤, at least one <)."""
        at_least_one_better = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                at_least_one_better = True
        return at_least_one_better

    def _crowding_distance_all(
        self, fronts: List[List[int]], objectives: List[Objectives]
    ) -> List[float]:
        """Compute crowding distance for all individuals."""
        n = len(objectives)
        crowd = [0.0] * n
        n_obj = len(objectives[0]) if objectives else 3

        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    crowd[idx] = float("inf")
                continue

            for m in range(n_obj):
                sorted_front = sorted(front, key=lambda i: objectives[i][m])
                crowd[sorted_front[0]] = float("inf")
                crowd[sorted_front[-1]] = float("inf")

                obj_range = (
                    objectives[sorted_front[-1]][m] - objectives[sorted_front[0]][m]
                )
                if obj_range == 0:
                    continue

                for k in range(1, len(sorted_front) - 1):
                    crowd[sorted_front[k]] += (
                        objectives[sorted_front[k + 1]][m]
                        - objectives[sorted_front[k - 1]][m]
                    ) / obj_range

        return crowd

    # ── Offspring generation ──────────────────────────────────────────────────

    def _make_offspring(
        self,
        population: List[DecoderTree],
        fronts: List[List[int]],
        crowd: List[float],
    ) -> List[DecoderTree]:
        """Tournament selection → crossover → mutation."""
        offspring = []
        target = self.config.pop_size

        while len(offspring) < target:
            p1 = self._tournament_select(population, fronts, crowd)
            p2 = self._tournament_select(population, fronts, crowd)

            if random.random() < self.config.crossover_rate:
                c1, c2 = TreeOperator.crossover(p1, p2)
            else:
                c1, c2 = p1, p2

            c1 = TreeOperator.mutate(c1, self.config.mutation_rate)
            c2 = TreeOperator.mutate(c2, self.config.mutation_rate)

            offspring.extend([c1, c2])

        return offspring[:target]

    def _tournament_select(
        self,
        population: List[DecoderTree],
        fronts: List[List[int]],
        crowd: List[float],
    ) -> DecoderTree:
        """Binary tournament on (front rank, -crowding distance)."""
        # Build rank map
        rank_map = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                rank_map[idx] = rank

        k = min(self.config.tournament_size, len(population))
        candidates = random.sample(range(len(population)), k)

        def key(i):
            return (rank_map.get(i, len(fronts)), -crowd[i])

        winner = min(candidates, key=key)
        return population[winner]

    # ── Environmental selection (elitism) ─────────────────────────────────────

    def _environmental_selection(
        self,
        combined: List[DecoderTree],
        objectives: List[Objectives],
    ) -> List[DecoderTree]:
        """Keep the best pop_size individuals from combined population."""
        fronts = self._non_dominated_sort(objectives)
        crowd = self._crowding_distance_all(fronts, objectives)

        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= self.config.pop_size:
                selected.extend(front)
            else:
                remaining = self.config.pop_size - len(selected)
                sorted_front = sorted(front, key=lambda i: -crowd[i])
                selected.extend(sorted_front[:remaining])
                break

        return [combined[i] for i in selected]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_generation(
        self,
        gen: int,
        arch_scores: List[ArchScore],
        fronts: List[List[int]],
        population: Optional[List[DecoderTree]] = None,
    ) -> None:
        n_pareto = len(fronts[0]) if fronts else 0
        avg_grad = sum(a.score_grad for a in arch_scores) / max(len(arch_scores), 1)
        avg_grad_obj = sum(a.score_grad_stability for a in arch_scores) / max(len(arch_scores), 1)

        pareto_set = set(fronts[0]) if fronts else set()

        logger.info(
            f"Gen {gen:3d} | Pareto={n_pareto}/{len(arch_scores)} | "
            f"Grad={avg_grad:.4e} | GradObj={avg_grad_obj:.4f}"
        )

        individuals = []
        for idx, sc in enumerate(arch_scores):
            tag = " [PARETO]" if idx in pareto_set else ""
            gene = ("  " + population[idx].summary()) if population and idx < len(population) else ""
            logger.info(
                f"  [{idx:3d}]{tag} "
                f"grad={sc.score_grad:.4e}  grad_obj={sc.score_grad_stability:.4f}  "
                f"flops={_format_flops(sc.flops)}  params={_format_params(sc.params)}"
                f"{gene}"
            )
            individuals.append({
                "idx": idx,
                "pareto": idx in pareto_set,
                "grad": sc.score_grad,
                "grad_obj": sc.score_grad_stability,
                "flops": sc.flops,
                "flops_g": sc.flops / 1e9,
                "params": sc.params,
                "params_m": sc.params / 1e6,
                "arch": population[idx].summary() if population and idx < len(population) else None,
            })

        self._gen_log.append({
            "gen": gen,
            "n_pareto": n_pareto,
            "avg_grad": avg_grad,
            "avg_grad_obj": avg_grad_obj,
            "individuals": individuals,
        })
        self._flush_history()

    def _flush_history(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._log_path.with_suffix(self._log_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._gen_log, f, indent=2)
        tmp_path.replace(self._log_path)

    def history(self) -> List[Dict]:
        return self._gen_log
