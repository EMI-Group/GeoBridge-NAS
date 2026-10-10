#!/usr/bin/env python3
"""Pre-cache dual-teacher + DINOv2 features to LMDB.

Run once before NAS search. Features are indexed by sample index so the
search loop can read them without re-running the heavy teacher models.

Single-GPU:
    python scripts/cache_features.py \
        --data_root /path/to/tartanair --cache_dir /path/to/cache \
        --scale_ckpt checkpoints/scale_teacher_best.pth

Multi-GPU DDP (recommended):
    torchrun --nproc_per_node=4 scripts/cache_features.py \
        --data_root /path/to/tartanair --cache_dir /path/to/cache \
        --scale_ckpt checkpoints/scale_teacher_best.pth

Each rank handles a contiguous slice of the dataset and writes to the same
LMDB (LMDB's file-lock serialises write transactions across processes).

Cache format (per sample):
    {
      "image":   (3, H, W)  float32 tensor
      "F_TA":    {"layer_2": (1024,), "layer_5": (1024,)}
      "F_TB":    {"layer_8": (C,),    "layer_11": (C,)}
    }
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.teachers.boundary_teacher import BoundaryTeacher, BOUNDARY_LAYERS
from ddn.teachers.scale_teacher import ScaleTeacher
from ddn.data.tartanair import TartanAirDataset
from ddn.data.feature_cache import FeatureCache
from ddn.proxy.distill_loss import SHALLOW_LAYERS, DEEP_LAYERS

# Boundary teacher layer → cache key
_BT_LAYER_KEYS = [f"layer_{l}" for l in BOUNDARY_LAYERS]
_BT_TO_CACHE   = dict(zip(_BT_LAYER_KEYS[:2], SHALLOW_LAYERS))

# Scale teacher stage → cache key
_ST_TO_CACHE = dict(zip(["stage_2", "stage_3"], DEEP_LAYERS))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--cache_dir",     required=True)
    p.add_argument("--boundary_ckpt", default=None)
    p.add_argument("--scale_ckpt",    required=True)
    p.add_argument("--dinov2_ckpt",   default=None)
    p.add_argument("--max_samples",   type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=4,
                   help="Per-GPU batch size.")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--split",         default="train")
    p.add_argument("--map_size_gb",   type=float, default=320)
    p.add_argument("--sparse_ratio",  type=float, default=0.05)
    p.add_argument("--img_h",         type=int, default=480)
    p.add_argument("--img_w",         type=int, default=640)
    p.add_argument("--log_every",     type=int, default=50,
                   help="Log progress every N batches.")
    p.add_argument("--resume",        action="store_true",
                   help="Skip already-cached indices (resume interrupted run).")
    return p.parse_args()


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def _pool(feat: torch.Tensor) -> torch.Tensor:
    """Global average pool: (B, C, H, W) → (B, C)."""
    if feat.dim() == 4:
        return feat.mean(dim=[2, 3])
    return feat


def main():
    args = parse_args()
    local_rank, world_size, is_ddp = setup_ddp()
    is_main = local_rank == 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank{local_rank}] %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_dataset = TartanAirDataset(
        root=args.data_root, split=args.split,
        sparse_ratio=args.sparse_ratio,
        max_samples=args.max_samples,
        img_size=(args.img_h, args.img_w),
    )

    n_total = len(full_dataset)
    if is_main:
        logger.info(f"Total samples: {n_total}, world_size: {world_size}")

    # Split dataset into contiguous per-rank slices (no padding/duplication)
    per_rank = (n_total + world_size - 1) // world_size
    rank_start = local_rank * per_rank
    rank_end   = min(rank_start + per_rank, n_total)
    rank_indices = list(range(rank_start, rank_end))

    # Resume: filter out already-cached indices before building the loader
    if args.resume and Path(args.cache_dir).exists():
        with FeatureCache(args.cache_dir, write=False) as _c:
            missing = [i for i in range(len(rank_indices)) if (rank_start + i) not in _c]
        n_done = len(rank_indices) - len(missing)
        logger.info(f"Resume: {n_done} already cached, {len(missing)} remaining")
        ds_indices     = [rank_indices[i] for i in missing]
        global_indices = [rank_start + i   for i in missing]
    else:
        ds_indices     = rank_indices
        global_indices = list(range(rank_start, rank_end))

    rank_dataset = Subset(full_dataset, ds_indices)
    loader = DataLoader(
        rank_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Handling samples [{rank_start}, {rank_end}) → {len(ds_indices)} to process")

    # ── Models ─────────────────────────────────────────────────────────────────
    boundary_teacher = BoundaryTeacher(
        pretrained=True,
        weight_path=args.boundary_ckpt,
        device=str(device),
    )
    scale_teacher = ScaleTeacher(
        checkpoint_path=args.scale_ckpt,
        device=str(device),
        freeze=True,
    )

    # ── Cache ──────────────────────────────────────────────────────────────────
    # All ranks write to the same LMDB; LMDB's write-lock serialises transactions.
    # set_mapsize() in FeatureCache.put() handles MDB_MAP_FULL automatically.
    sample_pos = 0
    try:
        with FeatureCache(args.cache_dir, write=True, map_size_gb=args.map_size_gb) as cache:
            for batch_no, batch in enumerate(loader):
                images       = batch["image"].to(device)
                sparse_depth = batch.get("sparse_depth")
                if sparse_depth is None:
                    sparse_depth = torch.zeros(
                        images.shape[0], 1, images.shape[2], images.shape[3], device=device
                    )
                else:
                    sparse_depth = sparse_depth.to(device)

                with torch.no_grad():
                    bt_feats = boundary_teacher.extract_features(images)
                    st_out   = scale_teacher(images, sparse_depth)

                F_TA = {
                    cache_key: _pool(bt_feats[bt_key])
                    for bt_key, cache_key in _BT_TO_CACHE.items()
                    if bt_key in bt_feats
                }
                F_TB = {
                    cache_key: _pool(st_out[st_key])
                    for st_key, cache_key in _ST_TO_CACHE.items()
                    if st_key in st_out
                }

                B = images.shape[0]
                for b in range(B):
                    cache.put(global_indices[sample_pos + b], {
                        "image": images[b].cpu(),
                        "F_TA":  {k: v[b].cpu() for k, v in F_TA.items()},
                        "F_TB":  {k: v[b].cpu() for k, v in F_TB.items()},
                    })
                sample_pos += B

                if batch_no % args.log_every == 0:
                    logger.info(
                        f"Progress: {sample_pos}/{len(ds_indices)} "
                        f"(global {global_indices[min(sample_pos, len(global_indices)-1)]}/{n_total})"
                    )

        logger.info(f"Done. Wrote {sample_pos} samples for rank {local_rank}")

        if is_ddp:
            dist.barrier()

        if is_main:
            logger.info(f"All ranks finished. Total cached: {n_total} samples → {args.cache_dir}")

    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
