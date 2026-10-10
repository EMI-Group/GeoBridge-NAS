#!/usr/bin/env python3
"""Train the Scale Teacher (ResNet-34 + NLSPN UNet) on TartanAir.

Single-GPU:
    python scripts/train_teacher.py \
        --data_root /path/to/tartanair --output_dir checkpoints/

Multi-GPU DDP (recommended):
    torchrun --nproc_per_node=4 scripts/train_teacher.py \
        --data_root /path/to/tartanair --output_dir checkpoints/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.teachers.scale_teacher import ScaleTeacher
from ddn.data.tartanair import TartanAirDataset
from ddn.data.kitti import KITTIEigenDataset
from ddn.train.losses import SILogLoss
from ddn.train.schedulers import warmup_cosine_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      choices=["tartanair", "kitti"], default="tartanair")
    p.add_argument("--data_root",    required=True, help="Dataset root")
    p.add_argument("--gt_root",      default=None, help="Depth root for KITTI; defaults to data_root.")
    p.add_argument("--filenames_file", default=None, help="Train split file for KITTI.")
    p.add_argument("--filenames_file_eval", default=None, help="Validation split file for KITTI.")
    p.add_argument("--output_dir",   default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Per-GPU batch size.")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs",type=int,   default=2)
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--sparse_ratio", type=float, default=0.05)
    p.add_argument("--img_h",        type=int,   default=480)
    p.add_argument("--img_w",        type=int,   default=640)
    p.add_argument("--depth_scale",  type=float, default=None)
    p.add_argument("--max_depth",    type=float, default=80.0)
    p.add_argument("--do_kb_crop",   action="store_true", default=False)
    p.add_argument("--do_random_rotate", action="store_true", default=False)
    p.add_argument("--degree",       type=float, default=2.5)
    p.add_argument("--use_right",    action="store_true", default=False)
    p.add_argument("--save_every",   type=int,   default=5)
    p.add_argument("--log_every",    type=int,   default=1000,
                   help="Print loss every N batches (0 = disable).")
    p.add_argument("--log_file",     default=None,
                   help="Path to save log file. None → stdout only.")
    return p.parse_args()


def setup_ddp():
    """Initialize DDP if torchrun set the env vars; otherwise single-GPU."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def main():
    args = parse_args()
    if args.dataset == "kitti":
        if args.depth_scale is None:
            args.depth_scale = 256.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 352, 1216
    local_rank, world_size, is_ddp = setup_ddp()
    is_main = local_rank == 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Dataset ────────────────────────────────────────────────────────────────
    if args.dataset == "tartanair":
        train_ds = TartanAirDataset(
            root=args.data_root, split="train",
            sparse_ratio=args.sparse_ratio,
            max_samples=args.max_samples,
            img_size=(args.img_h, args.img_w),
        )
        val_ds = TartanAirDataset(
            root=args.data_root, split="val",
            sparse_ratio=args.sparse_ratio,
            img_size=(args.img_h, args.img_w),
        )
    elif args.dataset == "kitti":
        if not args.filenames_file or not args.filenames_file_eval:
            raise ValueError("--dataset kitti requires --filenames_file and --filenames_file_eval")
        train_ds = KITTIEigenDataset(
            root=args.data_root,
            gt_root=args.gt_root,
            filenames_file=args.filenames_file,
            split="train",
            sparse_ratio=args.sparse_ratio,
            max_samples=args.max_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            do_kb_crop=args.do_kb_crop,
            do_random_rotate=args.do_random_rotate,
            degree=args.degree,
            use_right=args.use_right,
        )
        val_ds = KITTIEigenDataset(
            root=args.data_root,
            gt_root=args.gt_root,
            filenames_file=args.filenames_file_eval,
            split="val",
            sparse_ratio=0.0,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            do_kb_crop=args.do_kb_crop,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if is_main:
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        logger.info(f"World size: {world_size}, effective batch: {args.batch_size * world_size}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False)  if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ScaleTeacher(
        checkpoint_path=None,
        device=str(device),
        freeze=False,
        pretrained_backbone=True,
    )
    model.train()

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    criterion = SILogLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, n_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # ── Training ───────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        window_loss = 0.0  # accumulator for the current log window

        for step, batch in enumerate(train_loader, 1):
            images       = batch["image"].to(device)
            gt_depth     = batch["depth"].to(device)
            sparse_depth = batch.get("sparse_depth")
            if sparse_depth is None:
                sparse_depth = torch.zeros_like(gt_depth[:, :1])
            sparse_depth = sparse_depth.to(device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                out = model(images, sparse_depth)
                depth_pred = out["depth"]
                if depth_pred.shape[2:] != gt_depth.shape[2:]:
                    depth_pred = F.interpolate(
                        depth_pred, size=gt_depth.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = criterion(depth_pred, gt_depth)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_val = loss.item()
            train_loss  += loss_val
            window_loss += loss_val

            if is_main and args.log_every > 0 and step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d} | step {step:6d}/{len(train_loader)} | "
                    f"loss={window_loss / args.log_every:.4f} | lr={lr_now:.2e}"
                )
                window_loss = 0.0

        scheduler.step()

        # ── Validation (all ranks, average loss across GPUs) ──────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images       = batch["image"].to(device)
                gt_depth     = batch["depth"].to(device)
                sparse_depth = batch.get("sparse_depth")
                if sparse_depth is None:
                    sparse_depth = torch.zeros_like(gt_depth[:, :1])
                sparse_depth = sparse_depth.to(device)

                out = model(images, sparse_depth)
                depth_pred = out["depth"]
                if depth_pred.shape[2:] != gt_depth.shape[2:]:
                    depth_pred = F.interpolate(
                        depth_pred, size=gt_depth.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                val_loss += criterion(depth_pred, gt_depth).item()

        # Aggregate val loss across ranks so rank-0 logs the true average
        if is_ddp:
            val_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            avg_val = val_tensor.item() / (len(val_loader) * world_size)
        else:
            avg_val = val_loss / max(len(val_loader), 1)

        avg_train = train_loss / max(len(train_loader), 1)

        if is_main:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f}"
            )

            raw_model = model.module if is_ddp else model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(
                    {"model_state_dict": raw_model.state_dict(),
                     "epoch": epoch, "val_loss": avg_val},
                    out_dir / "scale_teacher_best.pth",
                )

            if epoch % args.save_every == 0:
                torch.save(
                    {"model_state_dict": raw_model.state_dict(), "epoch": epoch},
                    out_dir / f"scale_teacher_epoch{epoch:03d}.pth",
                )

    if is_main:
        logger.info(f"Training done. Best val_loss={best_val_loss:.4f}")
        logger.info(f"Best checkpoint: {out_dir / 'scale_teacher_best.pth'}")

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
