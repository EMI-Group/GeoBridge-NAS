#!/usr/bin/env python3
"""Precompute DepthAnything-V2 pseudo-depth (disparity) for RealEstate10K.

Runs the frozen DA-V2 ViT-L teacher ONCE over every frame at native ~518 res
and writes the raw disparity (DA's relu output, larger = closer) to disk as
fp16. The retrain script then reads these labels from disk instead of running
the teacher in the training loop — this isolates the teacher path (you can look
at the saved maps and confirm they are sharp), removes teacher forward cost from
training, and lets the student train at high resolution cheaply.

Storage layout mirrors the dataset chunks:

    <out_dir>/<split>/<chunk_stem>.torch  ->  {scene_idx: fp16 tensor [N, Hs, Ws]}

Alignment is by (chunk_stem, scene_idx, frame_idx) in the ORIGINAL (unshuffled)
chunk order — NOT by the scene "key" field (which may be absent). The retrain
dataset looks labels up the same way.

IMPORTANT: store aspect ratio should match the student training aspect
(--store_h/--store_w). The dataset force-resizes frames to (img_h, img_w), so if
you train square 448x448, store square 518x518 (default).

Multi-GPU: launch with torchrun; chunks are sharded across ranks by index.

Example:
    torchrun --nproc_per_node=4 scripts/prepare_pseudo_depth.py \
        --data_root /path/to/datasets/RealEstate10k/re10k \
        --split train \
        --boundary_ckpt checkpoints/depth_anything_v2_vitl.pth \
        --store_h 518 --store_w 518 \
        --out_dir /path/to/datasets/re10k_da_pseudo \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.data.re10k import Re10KFrameDataset
from ddn.teachers.boundary_teacher import BoundaryTeacher

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="ReSplat-style Re10K root containing <split>/*.torch")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True, help="Where to write <split>/<chunk_stem>.torch label files")
    p.add_argument("--boundary_ckpt", default=None, help="DepthAnything V2 ViT-L checkpoint")
    p.add_argument("--store_h", type=int, default=518, help="Stored disparity height (multiple of 14; match train aspect)")
    p.add_argument("--store_w", type=int, default=518, help="Stored disparity width (multiple of 14; match train aspect)")
    p.add_argument("--batch_size", type=int, default=16, help="Frames per DA forward pass")
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Recompute chunks even if the output file exists")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def collect_chunks(data_root: str, split: str, max_chunks):
    # Reuse the dataset's chunk discovery (handles index.json + glob).
    ds = Re10KFrameDataset.__new__(Re10KFrameDataset)
    ds.roots = [Path(data_root)]
    ds.split = split
    ds.max_chunks = max_chunks
    chunks = ds._collect_chunks()
    return chunks


@torch.no_grad()
def process_chunk(chunk_path, teacher, transform, store_hw, batch_size, device):
    """Return {scene_idx: fp16 disparity tensor [N, Hs, Ws]} for one chunk."""
    scenes = Re10KFrameDataset._load_chunk(chunk_path)
    out = {}
    for scene_idx, scene in enumerate(scenes):
        images = list(scene["images"])
        if not images:
            out[scene_idx] = torch.empty(0, *store_hw, dtype=torch.float16)
            continue
        tensors = []
        for encoded in images:
            img = Re10KFrameDataset._decode_image(encoded)
            tensors.append(transform(img))
        frames = torch.stack(tensors)  # (N, 3, Hs, Ws)

        disps = []
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size].to(device, non_blocking=True)
            disp = teacher.predict_depth(batch)  # (B, 1, Hs, Ws) raw disparity
            disps.append(disp.squeeze(1).to(torch.float16).cpu())
        out[scene_idx] = torch.cat(disps, dim=0)  # (N, Hs, Ws) fp16
    return out


def main():
    args = parse_args()
    local_rank, world_size, is_ddp = setup_ddp()
    is_main = local_rank == 0
    device = (
        torch.device("cuda", local_rank)
        if is_ddp and torch.cuda.is_available()
        else torch.device(args.device)
    )

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    for dim in (args.store_h, args.store_w):
        if dim % 14 != 0:
            raise ValueError(f"--store_h/--store_w must be multiples of 14 (got {args.store_h}x{args.store_w})")

    transform = T.Compose(
        [
            T.Resize((args.store_h, args.store_w), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    teacher = BoundaryTeacher(pretrained=True, weight_path=args.boundary_ckpt, device=str(device))
    teacher.eval()

    chunks = collect_chunks(args.data_root, args.split, args.max_chunks)
    if not chunks:
        raise FileNotFoundError(f"No Re10K chunks under {args.data_root}/{args.split}")
    if is_main:
        logger.info(
            "Found %d chunks; store=%dx%d fp16; out=%s",
            len(chunks), args.store_h, args.store_w, args.out_dir,
        )

    out_split = Path(args.out_dir) / args.split
    out_split.mkdir(parents=True, exist_ok=True)

    my_chunks = [c for i, c in enumerate(chunks) if i % world_size == local_rank]
    store_hw = (args.store_h, args.store_w)
    done = 0
    for chunk_path in my_chunks:
        dst = out_split / f"{chunk_path.stem}.torch"
        if dst.exists() and not args.overwrite:
            done += 1
            continue
        labels = process_chunk(chunk_path, teacher, transform, store_hw, args.batch_size, device)
        tmp = dst.with_suffix(".torch.tmp")
        torch.save(labels, tmp)
        tmp.rename(dst)
        done += 1
        n_frames = sum(t.shape[0] for t in labels.values())
        logger.info(
            "[rank %d] %s: %d scenes, %d frames -> %s",
            local_rank, chunk_path.name, len(labels), n_frames, dst.name,
        )

    if is_ddp:
        dist.barrier()
    if is_main:
        logger.info("Done. This rank wrote/skipped %d chunks.", done)
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
