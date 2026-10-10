#!/usr/bin/env python3
"""Distill DDN depth on RealEstate10K frames using DepthAnything pseudo depth.

Example:
    python scripts/train_re10k_depth.py \
        --arch 's32:M1D8FBSXNK3|s16:M1D8FBSXNK3|s8:M1D16FSXNK3|s4:M1D8FSXNK3' \
        --data_root datasets/re10k \
        --boundary_ckpt checkpoints/depth_anything_v2_vitl.pth \
        --dinov2_ckpt ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
        --output_dir checkpoints/retrain_re10k_da

    torchrun --nproc_per_node=4 scripts/train_re10k_depth.py \
        --arch 's32:M1D8FBSXNK3|s16:M1D8FBSXNK3|s8:M1D16FSXNK3|s4:M1D8FSXNK3' \
        --data_root datasets/re10k \
        --boundary_ckpt checkpoints/depth_anything_v2_vitl.pth \
        --dinov2_ckpt ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
        --batch_size 4 \
        --output_dir checkpoints/retrain_re10k_da_ddp

The saved ``model_state_dict`` is a StudentNet checkpoint and can be passed to
ReSplat via ``model.encoder.ddn_checkpoint_path=<checkpoint>``.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.data.re10k import Re10KFrameDataset, Re10KSceneGroupDataset
from ddn.search_space.tree import DecoderTree
from ddn.student.student_net import StudentNet
from ddn.teachers.boundary_teacher import BoundaryTeacher
from ddn.train.losses import MiDaSSSILoss, ScaleShiftInvariantDepthLoss, GroupSSILoss


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", "--genotype", dest="arch", default=None)
    p.add_argument("--arch_path", default=None, help="pareto_front.pkl from script 04")
    p.add_argument("--arch_idx", type=int, default=0)
    p.add_argument("--data_root", required=True, help="ReSplat-style Re10K root containing train/*.torch")
    p.add_argument("--split", default="train")
    p.add_argument("--val_data_root", default=None, help="Validation Re10K root. Defaults to --data_root.")
    p.add_argument("--val_split", default="test", help="Validation split under val_data_root.")
    p.add_argument("--val_every_steps", type=int, default=1000, help="Run validation every N global steps; 0 disables.")
    p.add_argument("--val_batches", type=int, default=100, help="Number of validation batches per validation pass.")
    p.add_argument("--val_max_chunks", type=int, default=None)
    p.add_argument("--val_max_frames", type=int, default=None)
    p.add_argument("--output_dir", default="checkpoints/retrain_re10k_da")
    p.add_argument("--boundary_ckpt", default=None, help="DepthAnything V2 ViT-L checkpoint")
    p.add_argument(
        "--pseudo_depth_dir",
        default=None,
        help="Directory of precomputed DA pseudo-depth (from scripts/prepare_pseudo_depth.py). When set, "
        "labels are read from disk and the live teacher is NOT loaded — saves VRAM/time "
        "and lets the student train at high resolution. See teacher_semantics note: the "
        "precompute stores raw disparity, so use --teacher_semantics depth here.",
    )
    p.add_argument(
        "--mvc_pseudo_dir",
        default=None,
        help="Directory of multi-view-CONSISTENT pseudo-depth (from scripts/prepare_multiview_depth.py). When set, "
        "training uses scene-GROUPED sampling + the per-scene GroupSSILoss so cross-frame "
        "scale consistency reaches the gradient. --batch_size counts GROUPS (effective frames "
        "= batch_size * group_size). Overrides --pseudo_depth_dir. Live teacher NOT loaded.",
    )
    p.add_argument("--group_size", type=int, default=4, help="frames per scene group (MVC mode)")
    p.add_argument("--require_accept", action=argparse.BooleanOptionalAction, default=True,
                   help="MVC mode: train only on scenes whose alignment passed the confidence gate")
    p.add_argument("--dinov2_ckpt", default=None, help="DINOv2 ViT-S checkpoint for StudentNet")
    p.add_argument("--resume", default=None, help="Resume a StudentNet checkpoint")
    p.add_argument(
        "--resume_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also restore optimizer/scheduler/global_step from --resume. "
             "Default keeps the historical behavior: load model weights only.",
    )

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--decoder_channels", type=int, default=256)

    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_encoder_bias", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_grad", type=float, default=1.0)
    p.add_argument(
        "--loss",
        choices=["midas", "ssi"],
        default="midas",
        help="midas: robust SSI + multi-scale gradient (sharper). ssi: legacy z-score loss.",
    )
    p.add_argument("--grad_scales", type=int, default=4, help="num scales for MiDaS gradient matching")
    p.add_argument("--trim", type=float, default=0.2, help="fraction of largest residuals dropped (MiDaS ssitrim)")
    p.add_argument(
        "--teacher_input_size",
        type=int,
        default=518,
        help="Run the DA teacher at this resolution (native 518) for sharp labels, then "
        "resize the pseudo-depth back to the training resolution. 0 = use training res.",
    )
    p.add_argument(
        "--unfreeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Full fine-tune the DINOv2 encoder instead of BitFit (bias-only). "
             "Defaults to True when --encoder_init da_vits, else False.",
    )
    p.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze every encoder parameter and optimize only scale_proj + decoder. "
             "This is safer for small scene-adaptation datasets.",
    )
    p.add_argument(
        "--encoder_init",
        choices=["raw", "da_vits"],
        default="raw",
        help="raw = DINOv2 self-supervised weights; da_vits = Depth-Anything-V2 "
             "ViT-S depth-tuned backbone (sharper depth tokens). With da_vits, "
             "--dinov2_ckpt must point at a Depth-Anything-V2 ViT-S checkpoint "
             "(its 'pretrained.*' backbone is extracted automatically), and the "
             "encoder is unfrozen by default.",
    )
    p.add_argument("--lr_encoder", type=float, default=2e-5, help="LR for the encoder when --unfreeze_encoder")
    p.add_argument(
        "--teacher_semantics",
        choices=["inverse_depth", "depth"],
        default="inverse_depth",
        help="DepthAnything relative output is commonly inverse-depth-like; invert it before distilling by default.",
    )
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--cuda_cleanup_every_steps",
        type=int,
        default=1000,
        help="Run gc.collect() and torch.cuda.empty_cache() every N train steps; 0 disables.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--ddp_backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend used under torchrun. auto -> nccl on CUDA, otherwise gloo.",
    )
    p.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DDP unused-parameter detection if experimenting with partial heads.",
    )
    return p.parse_args()


def setup_ddp(args):
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False, None

    backend = args.ddp_backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True, backend


def cleanup_ddp(is_ddp: bool) -> None:
    if is_ddp:
        dist.destroy_process_group()


def reduce_mean(metrics: Dict[str, float], device: torch.device, is_ddp: bool) -> Dict[str, float]:
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    if is_ddp:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {k: values[i].item() for i, k in enumerate(keys)}


def load_arch(args) -> DecoderTree:
    if args.arch:
        return DecoderTree.from_summary(args.arch, channels=args.decoder_channels)
    if not args.arch_path:
        raise ValueError("Pass either --arch '<genotype>' or --arch_path search_results/pareto_front.pkl")
    with open(args.arch_path, "rb") as f:
        pareto_front = pickle.load(f)
    if args.arch_idx >= len(pareto_front):
        raise IndexError(f"--arch_idx {args.arch_idx} out of range")
    arch = pareto_front[args.arch_idx]
    if arch.channels != args.decoder_channels:
        arch = DecoderTree.from_summary(arch.summary(), channels=args.decoder_channels)
    return arch


def build_loader(
    args,
    rank: int,
    *,
    roots: str,
    split: str,
    max_chunks: int | None,
    max_frames: int | None,
    shuffle: bool,
) -> DataLoader:
    if args.mvc_pseudo_dir:
        dataset = Re10KSceneGroupDataset(
            roots=roots,
            mvc_root=args.mvc_pseudo_dir,
            split=split,
            img_size=(args.img_h, args.img_w),
            group_size=args.group_size,
            max_chunks=max_chunks,
            require_accept=args.require_accept,
            shuffle=shuffle,
            seed=args.seed + rank,
        )
    else:
        dataset = Re10KFrameDataset(
            roots=roots,
            split=split,
            img_size=(args.img_h, args.img_w),
            max_chunks=max_chunks,
            max_frames=max_frames,
            shuffle=shuffle,
            seed=args.seed + rank,
            pseudo_depth_root=args.pseudo_depth_dir,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def unpack_batch(batch, device, args, teacher):
    """Return (images, target, groups). groups is None except in MVC mode, where
    a grouped batch [B,G,...] is flattened to [B*G,...] and groups maps each frame
    to its scene so GroupSSILoss can normalise per scene."""
    if args.mvc_pseudo_dir:
        imgs, tgt = batch["image"], batch["target"]          # [B,G,3,h,w], [B,G,1,Hs,Ws]
        B, G = imgs.shape[:2]
        imgs = imgs.flatten(0, 1).to(device, non_blocking=True)
        tgt = tgt.flatten(0, 1).to(device, non_blocking=True)
        groups = torch.arange(B, device=device).repeat_interleave(G)
        return imgs, tgt, groups
    imgs = batch["image"].to(device, non_blocking=True)
    if teacher is None:
        tgt = apply_semantics(batch["target"].to(device, non_blocking=True), args.teacher_semantics)
    else:
        tgt = teacher_target(teacher, imgs, args.teacher_semantics, args.teacher_input_size)
    return imgs, tgt, None


def run_criterion(criterion, pred, target, groups):
    return criterion(pred, target, groups) if groups is not None else criterion(pred, target)


def infinite(loader: DataLoader) -> Iterable[dict]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("Re10K dataloader produced no batches")


def teacher_target(
    teacher: BoundaryTeacher,
    images: torch.Tensor,
    semantics: str,
    teacher_size: int = 0,
) -> torch.Tensor:
    """Pseudo-depth label from DepthAnything.

    When ``teacher_size`` is set, the teacher is run at that (native, e.g. 518)
    resolution and the prediction is resized back to the training resolution, so
    labels keep DA's full-quality structure even when the student trains smaller.
    """
    with torch.no_grad():
        h, w = images.shape[-2:]
        timg = images
        if teacher_size and max(h, w) != teacher_size:
            s = teacher_size / max(h, w)
            nh = max(14, int(round(h * s / 14)) * 14)
            nw = max(14, int(round(w * s / 14)) * 14)
            timg = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
        target = teacher.predict_depth(timg).clamp_min(1e-6)
        if semantics == "inverse_depth":
            target = target.reciprocal()
        if target.shape[-2:] != (h, w):
            target = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
        return target


def apply_semantics(disparity: torch.Tensor, semantics: str) -> torch.Tensor:
    """Map a stored raw DA disparity label to the chosen training target.

    The precompute (scripts/prepare_pseudo_depth.py) stores DA's raw disparity (larger = closer).
    ``depth`` (recommended for pseudo-depth): use disparity as-is — no
    reciprocal, so far/sky pixels never blow up into outliers and the data
    term stays well-behaved. ``inverse_depth``: reciprocal to depth-like, a
    drop-in for the old live-teacher default (clamped to bound outliers).
    """
    if semantics == "inverse_depth":
        return disparity.clamp_min(1e-3).reciprocal()
    return disparity


@torch.no_grad()
def validate(
    model: nn.Module,
    teacher: BoundaryTeacher,
    loader: DataLoader,
    criterion: ScaleShiftInvariantDepthLoss,
    args,
    device: torch.device,
    is_ddp: bool,
) -> Dict[str, float]:
    model.eval()
    metrics = {"total": 0.0, "pixel": 0.0, "grad": 0.0}
    batches = infinite(loader)

    for _ in range(args.val_batches):
        batch = next(batches)
        images, target, groups = unpack_batch(batch, device, args, teacher)
        pred = model(images)["depth"]
        losses = run_criterion(criterion, pred, target, groups)
        for key in metrics:
            metrics[key] += losses[key].detach().item()

    return reduce_mean(
        {key: value / args.val_batches for key, value in metrics.items()},
        device,
        is_ddp,
    )


def load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def save_checkpoint(
    path: Path,
    student: StudentNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
    epoch: int,
    global_step: int,
    best_loss: float,
    best_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "best_metric": best_metric,
            "arch": args.arch,
            "args": vars(args),
        },
        path,
    )
    logger.info("Saved checkpoint: %s", path)


def main():
    args = parse_args()
    local_rank, world_size, is_ddp, ddp_backend = setup_ddp(args)
    is_main = local_rank == 0
    if is_ddp and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    torch.manual_seed(args.seed + local_rank)

    arch = load_arch(args)
    args.arch = arch.summary()
    if is_main:
        logger.info("World size: %s, per-GPU batch: %s", world_size, args.batch_size)
        if is_ddp:
            logger.info("DDP enabled: backend=%s", ddp_backend)
        logger.info("Architecture: %s", args.arch)

    # Resolve the encoder-init policy. da_vits = start the student encoder from
    # Depth-Anything-V2's depth-tuned ViT-S backbone (see DINOv2Encoder, which
    # strips the 'pretrained.' prefix), and full-finetune by default so the
    # depth-aware tokens can adapt.
    if args.encoder_init == "da_vits":
        if not args.dinov2_ckpt:
            raise ValueError("--encoder_init da_vits requires --dinov2_ckpt <Depth-Anything-V2 ViT-S .pth>")
        if args.unfreeze_encoder is None:
            args.unfreeze_encoder = True
    elif args.unfreeze_encoder is None:
        args.unfreeze_encoder = False
    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError("--freeze_encoder and --unfreeze_encoder are mutually exclusive")
    if is_main:
        logger.info("Encoder init: %s (weights=%s)", args.encoder_init, args.dinov2_ckpt)

    student = StudentNet(
        decoder_tree=arch,
        trainable_bias=False,
        decoder_channels=args.decoder_channels,
        weight_path=args.dinov2_ckpt,
    )
    student.enable_retrain_mode()
    if args.freeze_encoder:
        student.freeze_encoder()
        if is_main:
            logger.info("Encoder: FROZEN (decoder-only adaptation)")
    elif args.unfreeze_encoder:
        student.enable_full_finetune()
        if is_main:
            logger.info("Encoder: FULL fine-tune (lr=%.2e)", args.lr_encoder)
    elif is_main:
        logger.info("Encoder: BitFit (bias/LN/LayerScale only, lr=%.2e)", args.lr_encoder_bias)
    if args.resume:
        state = load_checkpoint(args.resume)
        student.load_state_dict(state.get("model_state_dict", state), strict=False)
        if is_main:
            mode = "training-state" if args.resume_state else "model-only"
            logger.info("Loaded student checkpoint: %s (%s)", args.resume, mode)
    else:
        state = None
    student = student.to(device)

    if args.mvc_pseudo_dir or args.pseudo_depth_dir:
        teacher = None
        if is_main:
            src = args.mvc_pseudo_dir or args.pseudo_depth_dir
            logger.info("%s: reading labels from %s (live teacher NOT loaded)",
                        "MVC pseudo-depth" if args.mvc_pseudo_dir else "Pseudo-depth", src)
    else:
        teacher = BoundaryTeacher(
            pretrained=True,
            weight_path=args.boundary_ckpt,
            device=str(device),
        )
        teacher.eval()

    model: nn.Module = student
    if is_ddp:
        # Full encoder fine-tune may leave a few ViT params unused by
        # get_intermediate_layers (e.g. the final norm); enable detection so DDP
        # doesn't error out.
        find_unused = args.ddp_find_unused_parameters or args.unfreeze_encoder
        model = nn.parallel.DistributedDataParallel(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    train_loader = build_loader(
        args,
        local_rank,
        roots=args.data_root,
        split=args.split,
        max_chunks=args.max_chunks,
        max_frames=args.max_frames,
        shuffle=True,
    )
    train_batches = infinite(train_loader)

    val_loader = None
    if args.val_every_steps > 0:
        val_loader = build_loader(
            args,
            local_rank,
            roots=args.val_data_root or args.data_root,
            split=args.val_split,
            max_chunks=args.val_max_chunks,
            max_frames=args.val_max_frames,
            shuffle=False,
        )

    if args.freeze_encoder:
        encoder_params = []
        encoder_lr = None
    elif args.unfreeze_encoder:
        encoder_params = student.encoder_finetune_parameters()
        encoder_lr = args.lr_encoder
    else:
        encoder_params = student.encoder_bias_parameters()
        encoder_lr = args.lr_encoder_bias
    parameter_groups = [
        {"params": student.decoder_parameters(), "lr": args.lr_decoder},
    ]
    max_lrs = [args.lr_decoder]
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": encoder_lr})
        max_lrs.append(encoder_lr)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lrs,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1
    if args.resume and args.resume_state and isinstance(state, dict):
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        best_val_loss = float(state.get("best_loss", best_val_loss))
        global_step = int(state.get("global_step", 0) or 0)
        start_epoch = int(state.get("epoch", 0) or 0) + 1
        if is_main:
            logger.info(
                "Restored training state: start_epoch=%d global_step=%d best_loss=%.6f",
                start_epoch,
                global_step,
                best_val_loss,
            )
    if args.mvc_pseudo_dir:
        # Per-SCENE scale-consistent loss: required so the multi-view consistency
        # in the pseudo-labels reaches the gradient (per-image SSI would erase it).
        criterion = GroupSSILoss(
            lambda_grad=args.lambda_grad, num_scales=args.grad_scales, trim=args.trim,
        )
    elif args.loss == "midas":
        criterion = MiDaSSSILoss(
            lambda_grad=args.lambda_grad,
            num_scales=args.grad_scales,
            trim=args.trim,
        )
    else:
        criterion = ScaleShiftInvariantDepthLoss(lambda_grad=args.lambda_grad)
    if is_main:
        loss_name = "group-ssi(MVC)" if args.mvc_pseudo_dir else args.loss
        logger.info(
            "Loss=%s lambda_grad=%.2f grad_scales=%d trim=%.2f group_size=%d",
            loss_name, args.lambda_grad, args.grad_scales, args.trim,
            args.group_size if args.mvc_pseudo_dir else 0,
        )
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    output_dir = Path(args.output_dir)

    if is_main:
        logger.info("Student params: %s", student.param_count())
        if val_loader is not None:
            logger.info(
                "Validation: split=%s every_steps=%d batches=%d",
                args.val_split,
                args.val_every_steps,
                args.val_batches,
            )
        logger.info("Starting DepthAnything distillation on Re10K")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = {"total": 0.0, "pixel": 0.0, "grad": 0.0}

        for step in range(1, args.steps_per_epoch + 1):
            global_step += 1
            batch = next(train_batches)
            images, target, groups = unpack_batch(batch, device, args, teacher)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                pred = model(images)["depth"]
                if not torch.isfinite(pred).all():
                    finite_fraction = torch.isfinite(pred).float().mean().item()
                    raise FloatingPointError(
                        f"Non-finite prediction at global_step={global_step}; "
                        f"finite_fraction={finite_fraction:.6f}"
                    )
                losses = run_criterion(criterion, pred, target, groups)
                loss = losses["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at global_step={global_step}: {loss.item()}"
                    )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at global_step={global_step}"
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at global_step={global_step}"
                    )
                optimizer.step()
            if global_step <= 5 or global_step % 10 == 0:
                for name, parameter in student.named_parameters():
                    if parameter.requires_grad and not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"Non-finite trainable parameter after global_step="
                            f"{global_step}: {name}"
                        )
            scheduler.step()

            for key in loss_sum:
                loss_sum[key] += losses[key].detach().item()

            if is_main and args.log_every > 0 and global_step % args.log_every == 0:
                logger.info(
                    "epoch=%d/%d step=%d/%d global_step=%d loss=%.4f pixel=%.4f grad=%.4f lr_dec=%.2e",
                    epoch,
                    args.epochs,
                    step,
                    args.steps_per_epoch,
                    global_step,
                    loss_sum["total"] / step,
                    loss_sum["pixel"] / step,
                    loss_sum["grad"] / step,
                    optimizer.param_groups[0]["lr"],
                )

            if (
                val_loader is not None
                and args.val_every_steps > 0
                and global_step % args.val_every_steps == 0
            ):
                val_metrics = validate(
                    model,
                    teacher,
                    val_loader,
                    criterion,
                    args,
                    device,
                    is_ddp,
                )
                model.train()
                if is_main:
                    logger.info(
                        "validation global_step=%d val_loss=%.4f val_pixel=%.4f val_grad=%.4f",
                        global_step,
                        val_metrics["total"],
                        val_metrics["pixel"],
                        val_metrics["grad"],
                    )
                    if val_metrics["total"] < best_val_loss:
                        best_val_loss = val_metrics["total"]
                        save_checkpoint(
                            output_dir / "best.pth",
                            student,
                            optimizer,
                            scheduler,
                            args,
                            epoch,
                            global_step,
                            best_val_loss,
                            "val_loss",
                        )
                    save_checkpoint(
                        output_dir / "latest.pth",
                        student,
                        optimizer,
                        scheduler,
                        args,
                        epoch,
                        global_step,
                        best_val_loss,
                        "val_loss",
                    )
                if is_ddp:
                    dist.barrier()
                cleanup_memory(device)

            del batch, images, target, groups, pred, losses, loss
            cleanup_every = max(0, int(args.cuda_cleanup_every_steps or 0))
            if cleanup_every and global_step % cleanup_every == 0:
                cleanup_memory(device)

        epoch_metrics = reduce_mean(
            {key: value / args.steps_per_epoch for key, value in loss_sum.items()},
            device,
            is_ddp,
        )
        if is_main:
            logger.info(
                "epoch=%d/%d train_loss=%.4f pixel=%.4f grad=%.4f",
                epoch,
                args.epochs,
                epoch_metrics["total"],
                epoch_metrics["pixel"],
                epoch_metrics["grad"],
            )
            if val_loader is None and epoch_metrics["total"] < best_val_loss:
                best_val_loss = epoch_metrics["total"]
                save_checkpoint(
                    output_dir / "best.pth",
                    student,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    global_step,
                    best_val_loss,
                    "train_loss",
                )
            if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
                save_checkpoint(
                    output_dir / f"epoch_{epoch:03d}.pth",
                    student,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    global_step,
                    best_val_loss,
                    "val_loss" if val_loader is not None else "train_loss",
                )
            save_checkpoint(
                output_dir / "latest.pth",
                student,
                optimizer,
                scheduler,
                args,
                epoch,
                global_step,
                best_val_loss,
                "val_loss" if val_loader is not None else "train_loss",
            )
        if is_ddp:
            dist.barrier()
        cleanup_memory(device)

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
