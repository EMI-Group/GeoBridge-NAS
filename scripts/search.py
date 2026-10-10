#!/usr/bin/env python3
"""NSGA-II NAS search for tree-coded Mamba decoder cells.

Reads pre-cached features (from cache_features.py) and runs NSGA-II to find
the Pareto front of fixed-trunk Mamba decoders that balance gradient stability,
FLOPs, and Params.

Single-GPU:
    python scripts/search.py \
        --cache_dir /path/to/feature_cache \
        --output_dir search_results/ \
        [--pop_size 64] [--n_generations 100]

Multi-GPU DDP (recommended for large populations):
    torchrun --nproc_per_node=4 scripts/search.py \
        --cache_dir /path/to/feature_cache \
        --output_dir search_results/ \
        [--pop_size 64] [--n_generations 100]

Each GPU evaluates a contiguous slice of the population per generation.
Rank 0 drives all NSGA-II selection / crossover / mutation logic.

Outputs:
    search_results/pareto_front.pkl        — list of DecoderTree objects
    search_results/pareto_summary.json     — lightweight JSON summary
    search_results/search_log.json         — generation-by-generation log
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.search.nsga2 import NSGA2, NSGA2Config
from ddn.search.fitness_eval import build_evaluator_from_cache


def setup_ddp():
    """Return (local_rank, world_size, is_ddp). No-op if not launched via torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",        required=True)
    p.add_argument("--output_dir",       default="search_results")
    p.add_argument("--pop_size",         type=int,   default=64)
    p.add_argument("--n_generations",    type=int,   default=100)
    p.add_argument("--decoder_channels", type=int,   default=256)
    p.add_argument("--n_eval_samples",   type=int,   default=512,
                   help="Total samples drawn from cache for fitness eval.")
    p.add_argument("--crossover_rate",   type=float, default=0.7)
    p.add_argument("--mutation_rate",    type=float, default=0.4)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--device",             default=None,
                   help="Device override (default: cuda:LOCAL_RANK or cuda).")
    p.add_argument("--forward_batch_size", type=int,   default=4,
                   help="Images per forward pass (reduce to save VRAM).")
    p.add_argument("--grad_batch_size",    type=int,   default=8,
                   help="Images used for grad-sensitivity backward pass.")
    p.add_argument("--dinov2_ckpt",        default=None)
    p.add_argument("--depth_target_path",  default=None,
                   help="Optional cached spatial pseudo-depth target .pt for GSAP.")
    return p.parse_args()


def main():
    local_rank, world_size, is_ddp = setup_ddp()
    is_main = local_rank == 0

    args = parse_args()
    random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    # Resolve device: DDP overrides any --device flag
    if is_ddp:
        device = f"cuda:{local_rank}"
    elif args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [rank{local_rank}] %(levelname)s %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample indices from cache ──────────────────────────────────────────────
    n_needed = args.grad_batch_size
    if args.n_eval_samples < n_needed:
        if is_main:
            logger.warning(
                f"--n_eval_samples {args.n_eval_samples} < required {n_needed}; "
                f"bumping to {n_needed}"
            )
        args.n_eval_samples = n_needed

    batch_indices = list(range(args.n_eval_samples))
    if is_main:
        logger.info(
            f"Using {len(batch_indices)} samples from cache: {args.cache_dir} | "
            f"world_size={world_size} | device={device}"
        )

    # ── Build per-rank evaluator ───────────────────────────────────────────────
    # Each rank loads the same cached features onto its own GPU.
    evaluator = build_evaluator_from_cache(
        cache_dir=args.cache_dir,
        batch_indices=batch_indices,
        device=device,
        decoder_channels=args.decoder_channels,
        forward_batch_size=args.forward_batch_size,
        grad_batch_size=args.grad_batch_size,
        dinov2_weight_path=args.dinov2_ckpt,
        depth_target_path=args.depth_target_path,
    )
    if is_main:
        logger.info("Gradient-stability FitnessEvaluator ready on all ranks.")

    # ── NSGA-II ────────────────────────────────────────────────────────────────
    cfg = NSGA2Config(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        decoder_channels=args.decoder_channels,
        seed=args.seed,
        log_path=out_dir / "search_log.json",
    )
    nsga = NSGA2(cfg)

    if is_main:
        logger.info(
            f"Starting NSGA-II: pop={cfg.pop_size}, gen={cfg.n_generations}, "
            f"ranks={world_size}"
        )

    if is_ddp:
        pareto_front = nsga.run_ddp(
            fitness_fn=evaluator,
            rank=local_rank,
            world_size=world_size,
        )
    else:
        pareto_front = nsga.run(fitness_fn=evaluator)

    # ── Save results (rank 0 only) ─────────────────────────────────────────────
    if is_main:
        logger.info(f"Search done. Pareto front size: {len(pareto_front)}")

        pkl_path = out_dir / "pareto_front.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pareto_front, f)
        logger.info(f"Saved Pareto front: {pkl_path}")

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
        with open(out_dir / "pareto_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # search_log.json is flushed after every logged generation during search.

        logger.info(f"All results saved to {out_dir}")
        logger.info("Next step: inspect pareto_summary.json, then run train_depth.py")

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
