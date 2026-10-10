#!/usr/bin/env python3
"""Evaluate teacher depth predictions on TartanAir.

NLSPN scale teacher:
    python scripts/evaluate_teacher.py \
        --teacher nlspn \
        --data_root /path/to/tartanair \
        --checkpoint checkpoints/scale_teacher_best.pth

DepthAnything v2 boundary teacher depth head:
    python scripts/evaluate_teacher.py \
        --teacher depthanything_v2 \
        --data_root /path/to/tartanair \
        --checkpoint checkpoints/depth_anything_v2_vitl.pth \
        --align median
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.data.tartanair import TartanAirDataset
from ddn.teachers.boundary_teacher import BoundaryTeacher
from ddn.teachers.scale_teacher import ScaleTeacher
from ddn.train.losses import SILogLoss
from ddn.utils.metrics import compute_depth_metrics


METRIC_KEYS = ["loss", "abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]
BOUNDARY_KEYS = ["boundary_precision", "boundary_recall", "boundary_fscore"]
ALL_METRIC_KEYS = METRIC_KEYS + BOUNDARY_KEYS
DEPTH_ANYTHING_PATCH_SIZE = 14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="nlspn", choices=["nlspn", "depthanything_v2"],
                   help="Teacher model to evaluate.")
    p.add_argument("--data_root", required=True, help="Dataset root")
    p.add_argument("--checkpoint", default=None,
                   help="Teacher checkpoint path. Defaults depend on --teacher.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=8, help="Per-process batch size.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sparse_ratio", type=float, default=0.05)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=100.0)
    p.add_argument("--align", default=None, choices=["none", "median"],
                   help="Depth scale alignment before metrics. Default: none for NLSPN, median for DepthAnything v2.")
    p.add_argument("--boundary_px", type=int, default=2,
                   help="Pixel tolerance radius for boundary matching.")
    p.add_argument("--boundary_quantile", type=float, default=0.90,
                   help="Gradient magnitude quantile used to binarize depth boundaries.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--pretrained_backbone", action="store_true",
                   help="Initialize backbone with ImageNet weights before loading checkpoint.")
    p.add_argument("--log_file", default=None)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def cleanup_dist(is_dist: bool):
    if is_dist:
        dist.destroy_process_group()


def reduce_totals(totals: Dict[str, float], count: int, device: torch.device, is_dist: bool):
    values = [totals[k] for k in ALL_METRIC_KEYS] + [float(count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if is_dist:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = {k: tensor[i].item() for i, k in enumerate(ALL_METRIC_KEYS)}
    reduced_count = int(tensor[-1].item())
    return reduced, reduced_count


def default_checkpoint(teacher: str) -> Path:
    if teacher == "nlspn":
        return Path("checkpoints/scale_teacher_best.pth")
    if teacher == "depthanything_v2":
        return Path("checkpoints/depth_anything_v2_vitl.pth")
    raise ValueError(f"Unknown teacher: {teacher}")


def build_teacher(args, checkpoint: Path, device: torch.device):
    if args.teacher == "nlspn":
        model = ScaleTeacher(
            checkpoint_path=str(checkpoint),
            device=str(device),
            freeze=True,
            pretrained_backbone=args.pretrained_backbone,
        )
        return model.eval()

    if args.teacher == "depthanything_v2":
        model = BoundaryTeacher(
            pretrained=True,
            weight_path=str(checkpoint),
            device=str(device),
        )
        return model.eval()

    raise ValueError(f"Unknown teacher: {args.teacher}")


def predict_depth(model, teacher: str, images: torch.Tensor, sparse_depth: torch.Tensor) -> torch.Tensor:
    if teacher == "nlspn":
        return model(images, sparse_depth)["depth"]
    if teacher == "depthanything_v2":
        _, _, height, width = images.shape
        pad_h = (DEPTH_ANYTHING_PATCH_SIZE - height % DEPTH_ANYTHING_PATCH_SIZE) % DEPTH_ANYTHING_PATCH_SIZE
        pad_w = (DEPTH_ANYTHING_PATCH_SIZE - width % DEPTH_ANYTHING_PATCH_SIZE) % DEPTH_ANYTHING_PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))

        pred = model.predict_depth(images)
        if pred.ndim == 3:
            pred = pred[:, :height, :width].unsqueeze(1)
        else:
            pred = pred[:, :, :height, :width]
        return pred
    raise ValueError(f"Unknown teacher: {teacher}")


def median_align_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    min_depth: float,
    max_depth: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    aligned = pred.clone()
    for i in range(pred.shape[0]):
        valid = (
            torch.isfinite(pred[i])
            & torch.isfinite(gt[i])
            & (pred[i] > eps)
            & (gt[i] > min_depth)
            & (gt[i] < max_depth)
        )
        if valid.any():
            scale = gt[i][valid].median() / pred[i][valid].median().clamp_min(eps)
            aligned[i] = pred[i] * scale
    return aligned


def compute_boundary_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    threshold_px: int,
    edge_quantile: float,
    min_depth: float,
    max_depth: float,
) -> Dict[str, float]:
    pred_edges = depth_edges(pred_depth, edge_quantile, min_depth, max_depth)
    gt_edges = depth_edges(gt_depth, edge_quantile, min_depth, max_depth)

    valid = ((gt_depth > min_depth) & (gt_depth < max_depth)).float()
    pred_edges = pred_edges * valid
    gt_edges = gt_edges * valid

    gt_dilated = dilate_binary(gt_edges, threshold_px)
    pred_dilated = dilate_binary(pred_edges, threshold_px)

    precision = (pred_edges * gt_dilated).sum(dim=(1, 2, 3)) / (
        pred_edges.sum(dim=(1, 2, 3)) + 1e-6
    )
    recall = (gt_edges * pred_dilated).sum(dim=(1, 2, 3)) / (
        gt_edges.sum(dim=(1, 2, 3)) + 1e-6
    )
    fscore = 2.0 * precision * recall / (precision + recall + 1e-6)

    return {
        "boundary_precision": precision.mean().item(),
        "boundary_recall": recall.mean().item(),
        "boundary_fscore": fscore.mean().item(),
    }


def depth_edges(
    depth: torch.Tensor,
    edge_quantile: float,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    depth = depth.clamp(min=min_depth, max=max_depth).float()
    log_depth = depth.clamp_min(1e-6).log()
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=log_depth.dtype,
        device=log_depth.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    mag = (
        F.conv2d(log_depth, sobel_x, padding=1) ** 2
        + F.conv2d(log_depth, sobel_y, padding=1) ** 2
    ).sqrt()

    valid = (depth > min_depth) & (depth < max_depth)
    edges = torch.zeros_like(mag)
    quantile = min(max(edge_quantile, 0.0), 1.0)
    for i in range(mag.shape[0]):
        vals = mag[i][valid[i]]
        if vals.numel() == 0:
            continue
        threshold = torch.quantile(vals.float(), quantile)
        edges[i] = (mag[i] >= threshold).float()
    return edges


def dilate_binary(binary: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return binary
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=binary.device, dtype=binary.dtype)
    return (F.conv2d(binary, kernel, padding=radius) > 0).float()


def main():
    args = parse_args()
    local_rank, world_size, is_dist = setup_dist()
    is_main = local_rank == 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    handlers = [logging.StreamHandler()]
    if is_main and args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    if args.align is None:
        args.align = "median" if args.teacher == "depthanything_v2" else "none"

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.teacher)
    if args.teacher == "nlspn" and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset = TartanAirDataset(
        root=args.data_root,
        split=args.split,
        sparse_ratio=args.sparse_ratio,
        max_samples=args.max_samples,
        img_size=(args.img_h, args.img_w),
    )

    if is_dist:
        dataset = Subset(dataset, range(local_rank, len(dataset), world_size))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_teacher(args, checkpoint, device)
    criterion = SILogLoss()

    totals = {k: 0.0 for k in ALL_METRIC_KEYS}
    seen = 0
    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            gt_depth = batch["depth"].to(device, non_blocking=True)
            sparse_depth = batch.get("sparse_depth")
            if sparse_depth is None:
                sparse_depth = torch.zeros_like(gt_depth[:, :1])
            sparse_depth = sparse_depth.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                depth_pred = predict_depth(model, args.teacher, images, sparse_depth)
                if depth_pred.shape[2:] != gt_depth.shape[2:]:
                    depth_pred = F.interpolate(
                        depth_pred,
                        size=gt_depth.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if args.align == "median":
                    depth_pred = median_align_depth(
                        depth_pred,
                        gt_depth,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
                loss = criterion(depth_pred, gt_depth)

            metrics = compute_depth_metrics(
                depth_pred,
                gt_depth,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            boundary_metrics = compute_boundary_metrics(
                depth_pred,
                gt_depth,
                threshold_px=args.boundary_px,
                edge_quantile=args.boundary_quantile,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            batch_size = images.shape[0]
            totals["loss"] += loss.item() * batch_size
            for key in METRIC_KEYS[1:]:
                totals[key] += metrics[key] * batch_size
            for key in BOUNDARY_KEYS:
                totals[key] += boundary_metrics[key] * batch_size
            seen += batch_size

    reduced, reduced_count = reduce_totals(totals, seen, device, is_dist)
    elapsed = time.perf_counter() - start

    if is_main:
        denom = max(reduced_count, 1)
        results = {k: reduced[k] / denom for k in ALL_METRIC_KEYS}
        results.update({
            "samples": reduced_count,
            "teacher": args.teacher,
            "align": args.align,
            "split": args.split,
            "checkpoint": str(checkpoint),
            "throughput_fps": reduced_count / max(elapsed, 1e-9),
            "world_size": world_size,
        })

        logger.info("Evaluation results:")
        logger.info(
            "teacher={teacher} | align={align} | loss={loss:.4f} | abs_rel={abs_rel:.4f} | sq_rel={sq_rel:.4f} | "
            "rmse={rmse:.4f} | rmse_log={rmse_log:.4f} | "
            "d1={d1:.4f} | d2={d2:.4f} | d3={d3:.4f} | "
            "b_precision={boundary_precision:.4f} | b_recall={boundary_recall:.4f} | "
            "b_fscore={boundary_fscore:.4f} | fps={throughput_fps:.2f}".format(**results)
        )

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2) + "\n")
            logger.info(f"Wrote metrics: {out_path}")

    cleanup_dist(is_dist)


if __name__ == "__main__":
    main()
