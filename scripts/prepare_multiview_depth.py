#!/usr/bin/env python3
"""Full-dataset precompute of MULTI-VIEW-CONSISTENT pseudo-depth for RealEstate10K.

For each scene, the pipeline runs DepthAnything on spread keyframes, aligns their per-frame scales to the
shared pose gauge by triangulation, applies the confidence gate, and stores the
aligned (multi-view-consistent) metric depth. Pair with the per-scene
``GroupSSILoss`` + ``Re10KSceneGroupDataset`` so the consistency actually reaches
the gradient (a per-image SSI loss would erase it).

Output layout (read by Re10KSceneGroupDataset):
    <out_dir>/<split>/<chunk_stem>.torch
    = {scene_idx: {"frames": Long[K], "depth": fp16[K, S, S], "accept": bool}}
where S = --da_size. Degenerate scenes (no parallax) are skipped. Rejected-by-gate
scenes are stored with accept=False holding raw DA depth (use require_accept=True
in the loader to train only on confidently-consistent scenes).

Multi-GPU: launch with torchrun; chunks are sharded across ranks.

Example:
    torchrun --nproc_per_node=4 scripts/prepare_multiview_depth.py \
        --data_root /path/to/datasets/RealEstate10k/re10k --split train \
        --boundary_ckpt checkpoints/depth_anything_v2_vitl.pth \
        --da_size 518 --max_frames 24 --min_tri_angle 1.0 \
        --out_dir /path/to/datasets/re10k_mvc_pseudo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "depthsplat"))

from ddn.data.re10k import Re10KFrameDataset
from ddn.data.multiview_alignment import solve_scene
from ddn.teachers.boundary_teacher import BoundaryTeacher

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--boundary_ckpt", default=None)
    # alignment knobs (forwarded to solve_scene) -- keep names identical
    p.add_argument("--da_size", type=int, default=518, help="DA + stored depth resolution (mult of 14)")
    p.add_argument("--da_batch", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=24, help="keyframes per scene (spread across the clip)")
    p.add_argument("--spread", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pair_strides", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--matches_per_pair", type=int, default=300)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--match_size", type=int, default=384, help="SIFT detection resolution (smaller=faster)")
    p.add_argument("--sift_nfeatures", type=int, default=2000)
    p.add_argument("--match_workers", type=int, default=4, help="threads for per-frame SIFT detection")
    p.add_argument("--min_tri_angle", type=float, default=1.0)
    p.add_argument("--affine", action="store_true")
    p.add_argument("--max_scale_ratio", type=float, default=10.0)
    p.add_argument("--min_gain", type=float, default=0.05)
    p.add_argument("--store_rejected", action=argparse.BooleanOptionalAction, default=True,
                   help="also store gate-rejected scenes (raw DA depth, accept=False)")
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    lr = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
    return lr, dist.get_world_size(), True


@torch.no_grad()
def process_chunk(chunk_path, solve_scene, args, teacher, device, log_scenes=False):
    scenes = Re10KFrameDataset._load_chunk(chunk_path)
    out = {}
    n_ok = n_acc = n_deg = 0
    for sidx, scene in enumerate(scenes):
        t0 = time.time()
        try:
            m = solve_scene(scene, args, teacher, device, verbose=False)
        except Exception as e:
            logger.warning("scene %s/%d failed: %s", chunk_path.stem, sidx, e)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if log_scenes:
            logger.info("  %s scene %d/%d: %s in %.1fs",
                        chunk_path.stem, sidx + 1, len(scenes),
                        m["status"] + ("/accept" if m.get("accept") else ""), time.time() - t0)
        if m["status"] != "ok":
            n_deg += 1
            continue
        if not m["accept"] and not args.store_rejected:
            continue
        # aligned (or identity, if rejected) metric depth at the keyframes.
        a, bsh, depth = m["a"], m["bsh"], m["depth"]          # depth [K,S,S]
        K = depth.shape[0]
        aligned = torch.stack([
            (a[f] * depth[f] + bsh[f]).clamp_min(1e-6) for f in range(K)
        ]).to(torch.float16)
        out[sidx] = {
            "frames": torch.tensor(m["fidx"], dtype=torch.long),
            "depth": aligned,
            "accept": bool(m["accept"]),
        }
        n_ok += 1
        n_acc += int(m["accept"])
    return out, (n_ok, n_acc, n_deg)


def main():
    args = parse_args()
    lr, world, is_ddp = setup_ddp()
    is_main = lr == 0
    device = torch.device("cuda", lr) if is_ddp and torch.cuda.is_available() else torch.device(args.device)
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)

    if args.da_size % 14 != 0:
        raise ValueError(f"--da_size must be a multiple of 14 (got {args.da_size})")


    teacher = BoundaryTeacher(pretrained=True, weight_path=args.boundary_ckpt, device=str(device))
    teacher.eval()

    ds = Re10KFrameDataset.__new__(Re10KFrameDataset)
    ds.roots = [Path(args.data_root)]; ds.split = args.split; ds.max_chunks = args.max_chunks
    chunks = ds._collect_chunks()
    if not chunks:
        raise FileNotFoundError(f"No chunks under {args.data_root}/{args.split}")
    if is_main:
        logger.info("Found %d chunks; store=%dx%d; gate(min_gain=%.2f,ratio<=%.1f); out=%s",
                    len(chunks), args.da_size, args.da_size, args.min_gain, args.max_scale_ratio, args.out_dir)

    out_split = Path(args.out_dir) / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    my_chunks = [c for i, c in enumerate(chunks) if i % world == lr]

    tot_ok = tot_acc = tot_deg = 0
    for ci, chunk_path in enumerate(my_chunks):
        dst = out_split / f"{chunk_path.stem}.torch"
        if dst.exists() and not args.overwrite:
            continue
        # Log per-scene progress for the first couple of chunks so you can see
        # activity + per-scene timing instead of waiting blind for a whole chunk.
        records, (n_ok, n_acc, n_deg) = process_chunk(
            chunk_path, solve_scene, args, teacher, device, log_scenes=(is_main and ci < 2))
        tmp = dst.with_suffix(".torch.tmp")
        torch.save(records, tmp); tmp.rename(dst)
        tot_ok += n_ok; tot_acc += n_acc; tot_deg += n_deg
        logger.info("[rank %d] %s: stored %d scenes (%d accepted, %d degenerate-skipped) [%d/%d]",
                    lr, chunk_path.name, n_ok, n_acc, n_deg, ci + 1, len(my_chunks))

    if is_ddp:
        dist.barrier(device_ids=[lr] if torch.cuda.is_available() else None)
    logger.info("[rank %d] done: stored=%d accepted=%d degenerate=%d", lr, tot_ok, tot_acc, tot_deg)
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
