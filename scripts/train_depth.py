#!/usr/bin/env python3
"""Retrain a selected architecture.

The architecture can come from a Pareto-front pickle written by script 04, or
from a direct genotype string such as:
    s32:M1D8FBSXNK3|s16:M1D8FBSXNK3|s8:M1D16FSXNK3|s4:M1D8FSXNK3

Usage:
    python scripts/train_depth.py \
        --arch_path search_results/pareto_front.pkl \
        --arch_idx 0 \
        --dataset nyu \
        --data_root /path/to/nyu \
        --filenames_file /path/to/nyu_train.txt \
        --filenames_file_eval /path/to/nyu_val.txt \
        --output_dir checkpoints/retrain/ \
        [--boundary_ckpt checkpoints/depth_anything_v2_vitl.pth] \
        [--scale_ckpt checkpoints/scale_teacher_best.pth] \
        [--epochs 50] \
        [--batch_size 8]

    torchrun --nproc_per_node=4 scripts/train_depth.py \
        --arch 's32:M1D8FBSXNK3|s16:M1D8FBSXNK3|s8:M1D16FSXNK3|s4:M1D8FSXNK3' \
        --dataset nyu \
        --data_root /path/to/nyu \
        --filenames_file /path/to/nyu_train.txt \
        --filenames_file_eval /path/to/nyu_val.txt \
        --batch_size 4

The retrained checkpoint is saved to:
    <output_dir>/epoch_<N>.pth   (every --save_every epochs)
"""

import argparse
import inspect
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.search_space.tree import DecoderTree
from ddn.student.student_net import StudentNet
from ddn.teachers.boundary_teacher import BoundaryTeacher
from ddn.teachers.scale_teacher import ScaleTeacher
from ddn.data.nyu import NYUDepthV2Dataset
from ddn.data.matterport3d import Matterport3DDataset
from ddn.data.tartanair import TartanAirDataset
from ddn.data.ddad import DDADDataset
from ddn.data.kitti import KITTIEigenDataset
from ddn.train.retrain import Retrainer, RetrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", "--genotype", dest="arch", default=None,
                   help="Direct genotype string, e.g. s32:...|s16:...|s8:...|s4:...")
    p.add_argument("--arch_path",       default=None, help="pareto_front.pkl from script 04")
    p.add_argument("--arch_idx",        type=int, default=0,
                   help="Which Pareto-front arch to retrain (0 = first)")
    p.add_argument("--decoder_baseline", choices=["tree", "vmamba", "localmamba"], default="tree",
                   help="Fixed decoder cell family. External options retain the DDN scaffold.")
    p.add_argument("--seed",            type=int, default=42,
                   help="Random seed for reproducible architecture retraining.")
    p.add_argument("--dataset",         choices=["nyu", "tartanair", "matterport3d", "ddad", "kitti"], default="nyu")
    p.add_argument("--data_root",       required=True, help="Dataset root")
    p.add_argument("--gt_root",         default=None, help="Depth root for DDAD; defaults to data_root.")
    p.add_argument("--val_data_root",   default=None, help="Validation dataset root; defaults to data_root.")
    p.add_argument("--val_gt_root",     default=None, help="Validation depth root for DDAD; defaults to val_data_root/gt_root.")
    p.add_argument("--filenames_file",  default=None, help="Train split file.")
    p.add_argument("--filenames_file_eval", default=None, help="Validation split file.")
    p.add_argument("--intrinsics_file", default=None, help="DDAD intrinsics JSON.")
    p.add_argument("--output_dir",      default="checkpoints/retrain")
    p.add_argument("--resume",          default=None,
                   help="Path to a checkpoint (e.g. checkpoints/retrain_ddad/last.pth) to "
                        "resume from: restores model/optimizer/scheduler/global_step/best_d1 "
                        "and continues after the last completed epoch.")
    p.add_argument("--boundary_ckpt",   default=None)
    p.add_argument("--scale_ckpt",      default=None)
    p.add_argument("--dinov2_ckpt",     default=None)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="Stop after this many completed epochs without d1 improvement; disabled by default.")
    p.add_argument("--early_stop_min_epochs", type=int, default=0,
                   help="Number of completed burn-in epochs; patience starts on the next epoch "
                        "(default: 0).")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Cumulative absolute d1 gain from the last qualifying best required to reset patience.")
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--decoder_channels",type=int,   default=256)
    p.add_argument("--lr_decoder",      type=float, default=1e-4)
    p.add_argument("--lr_encoder_bias", type=float, default=1e-5)
    p.add_argument("--lambda_distill",  type=float, default=0.1)
    p.add_argument("--warmup_epochs",   type=int,   default=0)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--lr_schedule",     choices=["epoch", "step"], default="epoch")
    p.add_argument("--min_lr_ratio",    type=float, default=0.0,
                   help="Final LR as a ratio of each param group's base LR.")
    p.add_argument("--accum_steps",     type=int,   default=1,
                   help="Gradient accumulation steps. Effective batch = "
                        "batch_size * world_size * accum_steps.")
    p.add_argument("--save_every",      type=int,   default=10)
    p.add_argument("--log_every_batches", type=int, default=50,
                   help="Print train loss every N train batches globally (0 = disable).")
    p.add_argument("--val_every_batches", type=int, default=1000,
                   help="Run validation every N train batches globally (0 = epoch only).")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--max_samples",     type=int,   default=None)
    p.add_argument("--val_batch_size",  type=int,   default=None,
                   help="Eval batch size (no grad → can be much larger than "
                        "--batch_size). Defaults to --batch_size. Bigger = faster eval.")
    p.add_argument("--val_max_samples", type=int,   default=None,
                   help="Randomly subsample the val set (fixed seed) for in-training "
                        "monitoring, e.g. 7500. Re-eval the best checkpoint on the FULL "
                        "val set with script 06 for the reported number.")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp",             action="store_true", default=True)
    p.add_argument("--sync_bn",         action="store_true", default=False,
                   help="Convert BatchNorm layers to SyncBatchNorm in DDP training.")
    p.add_argument("--full_finetune",   action="store_true", default=False,
                   help="Unfreeze the entire DINOv2 encoder (full fine-tune) instead of "
                        "BitFit bias-only. Use a small --encoder_lr to avoid forgetting.")
    p.add_argument("--finetune_blocks", type=int, default=0,
                   help="Unfreeze only the last N DINOv2 blocks (+ final norm). Middle "
                        "ground between BitFit (0) and full fine-tune; takes precedence "
                        "over --full_finetune when >0. ViT-S has 12 blocks.")
    p.add_argument("--encoder_lr",      type=float, default=None,
                   help="LR for encoder params under --full_finetune "
                        "(defaults to --lr_encoder_bias, i.e. 1e-5).")
    p.add_argument("--sparse_ratio",    type=float, default=0.00)
    p.add_argument("--img_h",           type=int,   default=480)
    p.add_argument("--img_w",           type=int,   default=640)
    # DDAD-only: training resizes the full frame to (ddad_resize_h, ddad_resize_w)
    # then random-crops to (ddad_train_h, ddad_train_w); eval uses (img_h, img_w).
    p.add_argument("--ddad_train_h",    type=int,   default=None)
    p.add_argument("--ddad_train_w",    type=int,   default=None)
    p.add_argument("--ddad_resize_h",   type=int,   default=None)
    p.add_argument("--ddad_resize_w",   type=int,   default=None)
    # KITTI-only: training random-crops the kb-cropped 352x1216 frame to
    # (kitti_train_h, kitti_train_w); eval runs on the full 352x1216 kb crop.
    p.add_argument("--kitti_train_h",   type=int,   default=None)
    p.add_argument("--kitti_train_w",   type=int,   default=None)
    p.add_argument("--kitti_use_right", action="store_true", default=False,
                   help="Randomly swap in image_03 (right camera) train frames. "
                        "Requires image_03 RGB + GT on disk.")
    p.add_argument("--depth_scale",     type=float, default=None)
    p.add_argument("--max_depth",       type=float, default=None)
    p.add_argument("--min_depth_eval",  type=float, default=1e-3)
    p.add_argument("--max_depth_eval",  type=float, default=None)
    p.add_argument("--eigen_crop",      action="store_true", default=False)
    p.add_argument("--garg_crop",       action="store_true", default=False)
    p.add_argument("--post_process_eval", action="store_true", default=False)
    return p.parse_args()


def setup_ddp():
    """Initialize DDP if torchrun set the env vars; otherwise single process."""
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


def set_random_seed(seed: int) -> None:
    """Seed all stochastic sources before constructing the decoder and loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # This makes the three-seed baseline comparison repeatable without forcing
    # strict deterministic algorithms that some selective-scan kernels may not support.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_arch(args):
    if args.arch:
        if args.arch.startswith("traditional_fpn("):
            from ddn.search_space.traditional import TraditionalDecoderArch
            arch = TraditionalDecoderArch.from_summary(args.arch, channels=args.decoder_channels)
        else:
            arch = DecoderTree.from_summary(args.arch, channels=args.decoder_channels)
        logger.info(
            f"Loaded arch ({type(arch).__name__}): params={arch.params():.2e}"
        )
        return arch

    if not args.arch_path:
        raise ValueError("Pass either --arch '<genotype>' or --arch_path search_results/pareto_front.pkl")

    arch_path = args.arch_path
    arch_idx = args.arch_idx
    with open(arch_path, "rb") as f:
        pareto_front = pickle.load(f)
    if arch_idx >= len(pareto_front):
        raise IndexError(
            f"--arch_idx {arch_idx} out of range (Pareto front has {len(pareto_front)} archs)"
        )
    arch = pareto_front[arch_idx]
    logger.info(
        f"Loaded arch #{arch_idx}: flops={arch.flops(args.img_h,args.img_w):.2e}, "
        f"params={arch.params():.2e}"
    )
    return arch


def main():
    args = parse_args()
    if args.dataset == "ddad":
        # PixelFormer DDAD protocol (matches ddad.zip arg file) for a fair
        # comparison against the PixelFormer baseline: full-frame resize (no
        # sky/hood crop), 200 m ceiling, NO garg/eigen eval crop, flip
        # post-process at eval, depth = metres * 256. Training resizes the full
        # frame to 608x960 then random-crops to 480x640; eval runs at 608x960.
        if args.depth_scale is None:
            args.depth_scale = 256.0
        if args.max_depth is None:
            args.max_depth = 200.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 608, 960          # eval / val resolution
        if args.ddad_train_h is None or args.ddad_train_w is None:
            args.ddad_train_h, args.ddad_train_w = 480, 640   # train random-crop
        if args.ddad_resize_h is None or args.ddad_resize_w is None:
            args.ddad_resize_h, args.ddad_resize_w = 608, 960  # train working size
        args.garg_crop = False
        args.eigen_crop = False
        args.post_process_eval = True
    elif args.dataset == "kitti":
        # Standard Eigen-split protocol (NewCRFs): kb-crop to 352x1216,
        # 80 m ceiling, garg crop at eval, flip post-process, depth PNG / 256.
        # Training random-crops the kb crop to 352x1120.
        if args.depth_scale is None:
            args.depth_scale = 256.0
        if args.max_depth is None:
            args.max_depth = 80.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 352, 1216         # eval / val resolution
        if args.kitti_train_h is None or args.kitti_train_w is None:
            args.kitti_train_h, args.kitti_train_w = 352, 1120  # train random-crop
        args.garg_crop = True
        args.eigen_crop = False
        args.post_process_eval = True
    else:
        if args.depth_scale is None:
            args.depth_scale = 1000.0
        if args.max_depth is None:
            args.max_depth = 10.0
    local_rank, world_size, is_ddp = setup_ddp()
    is_main = local_rank == 0
    device = torch.device("cuda", local_rank) if is_ddp else torch.device(args.device)
    set_random_seed(args.seed)

    handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    if is_main:
        logger.info(f"World size: {world_size}, per-GPU batch: {args.batch_size}")
        logger.info(f"Random seed: {args.seed}")
        if args.dataset == "ddad" and not args.sync_bn and args.batch_size <= 4 and is_ddp:
            logger.warning(
                "DDAD: per-GPU batch_size=%d is small and --sync_bn is off; "
                "decoder BatchNorm statistics may be noisy.",
                args.batch_size,
            )

    # ── Architecture ───────────────────────────────────────────────────────────
    arch = load_arch(args) if args.decoder_baseline == "tree" else None
    if is_main:
        logger.info(f"Architecture: {arch.summary() if arch is not None else args.decoder_baseline + ' official defaults'}")

    # ── Student network ────────────────────────────────────────────────────────
    student = StudentNet(
        decoder_tree=arch,
        trainable_bias=False,           # Retrainer calls enable_retrain_mode()
        decoder_channels=args.decoder_channels,
        weight_path=args.dinov2_ckpt,
        min_depth=args.min_depth_eval,
        max_depth=args.max_depth,
        decoder_baseline=args.decoder_baseline,
    )
    # Retrainer re-applies the final encoder mode before DDP; set it here too so
    # the early param-count log is accurate.
    if args.finetune_blocks > 0:
        student.enable_partial_finetune(args.finetune_blocks)
        mode = f"partial fine-tune (last {args.finetune_blocks} blocks)"
    elif args.full_finetune:
        student.enable_full_finetune()
        mode = "full fine-tune"
    else:
        student.enable_retrain_mode()
        mode = "BitFit bias-only"
    if is_main:
        logger.info(f"Encoder mode: {mode} | student params: {student.param_count()}")

    # ── Teachers (optional) ────────────────────────────────────────────────────
    boundary_teacher = None
    scale_teacher = None

    use_live_distill = args.lambda_distill > 0 and args.boundary_ckpt and args.scale_ckpt

    if use_live_distill:
        try:
            boundary_teacher = BoundaryTeacher(
                pretrained=True,
                weight_path=args.boundary_ckpt,
                device=str(device),
            )
            if is_main:
                logger.info("BoundaryTeacher loaded for live distillation.")
        except FileNotFoundError as e:
            if is_main:
                logger.warning(f"BoundaryTeacher not available: {e}")

    if use_live_distill and Path(args.scale_ckpt).exists():
        scale_teacher = ScaleTeacher(
            checkpoint_path=args.scale_ckpt,
            device=str(device),
            freeze=True,
        )
        if is_main:
            logger.info("ScaleTeacher loaded for live distillation.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    if args.dataset == "nyu":
        if not args.filenames_file or not args.filenames_file_eval:
            raise ValueError("--dataset nyu requires --filenames_file and --filenames_file_eval")
        train_ds = NYUDepthV2Dataset(
            root=args.data_root,
            filenames_file=args.filenames_file,
            split="train",
            max_samples=args.max_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
        )
        val_ds = NYUDepthV2Dataset(
            root=args.data_root,
            filenames_file=args.filenames_file_eval,
            split="val",
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
        )
    elif args.dataset == "tartanair":
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
    elif args.dataset == "matterport3d":
        if not args.filenames_file or not args.filenames_file_eval:
            raise ValueError("--dataset matterport3d requires --filenames_file and --filenames_file_eval")
        train_ds = Matterport3DDataset(
            root=args.data_root,
            filenames_file=args.filenames_file,
            max_samples=args.max_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            sparse_ratio=args.sparse_ratio,
            require_intrinsics=False,
        )
        val_ds = Matterport3DDataset(
            root=args.data_root,
            filenames_file=args.filenames_file_eval,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            sparse_ratio=0.0,
            require_intrinsics=False,
        )
    elif args.dataset == "ddad":
        if not args.filenames_file or not args.filenames_file_eval:
            raise ValueError("--dataset ddad requires --filenames_file and --filenames_file_eval")
        val_data_root = args.val_data_root or args.data_root
        val_gt_root = args.val_gt_root
        if val_gt_root is None:
            val_gt_root = args.val_data_root or args.gt_root
        train_ds = DDADDataset(
            root=args.data_root,
            gt_root=args.gt_root,
            intrinsics_file=args.intrinsics_file,
            filenames_file=args.filenames_file,
            split="train",
            max_samples=args.max_samples,
            img_size=(args.ddad_train_h, args.ddad_train_w),
            train_resize=(args.ddad_resize_h, args.ddad_resize_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            sparse_ratio=args.sparse_ratio,
        )
        val_ds = DDADDataset(
            root=val_data_root,
            gt_root=val_gt_root,
            intrinsics_file=args.intrinsics_file,
            filenames_file=args.filenames_file_eval,
            split="val",
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            sparse_ratio=0.0,
        )
    elif args.dataset == "kitti":
        if not args.filenames_file or not args.filenames_file_eval:
            raise ValueError("--dataset kitti requires --filenames_file and --filenames_file_eval")
        train_ds = KITTIEigenDataset(
            root=args.data_root,
            gt_root=args.gt_root,
            filenames_file=args.filenames_file,
            split="train",
            max_samples=args.max_samples,
            img_size=(args.kitti_train_h, args.kitti_train_w),
            depth_scale=args.depth_scale,
            min_depth=1.0,
            max_depth=args.max_depth,
            sparse_ratio=args.sparse_ratio,
            do_random_rotate=True,
            degree=1.0,
            use_right=args.kitti_use_right,
        )
        val_ds = KITTIEigenDataset(
            root=args.val_data_root or args.data_root,
            gt_root=args.val_gt_root or args.gt_root,
            filenames_file=args.filenames_file_eval,
            split="val",
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            sparse_ratio=0.0,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Randomly subsample the val set for fast in-training monitoring. Fixed seed
    # so the SAME subset is used at every validation step (stable d1 curve);
    # re-eval the best checkpoint on the FULL val set with script 06 to report.
    if args.val_max_samples is not None and len(val_ds) > args.val_max_samples:
        g = torch.Generator().manual_seed(0)
        subset_idx = torch.randperm(len(val_ds), generator=g)[:args.val_max_samples].tolist()
        val_ds = Subset(val_ds, subset_idx)
        if is_main:
            logger.info(f"Val monitoring subset: {len(val_ds)} random samples (seed=0)")
    if is_main:
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size or args.batch_size,
        sampler=val_sampler,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Retrain ────────────────────────────────────────────────────────────────
    cfg_kwargs = dict(
        n_epochs=args.epochs,
        lr_decoder=args.lr_decoder,
        lr_encoder_bias=args.lr_encoder_bias,
        batch_size=args.batch_size,
        lambda_distill=args.lambda_distill,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
        accum_steps=args.accum_steps,
        save_every=args.save_every,
        log_every_batches=args.log_every_batches,
        val_every_batches=args.val_every_batches,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval if args.max_depth_eval is not None else args.max_depth,
        eigen_crop=args.eigen_crop,
        garg_crop=args.garg_crop,
        post_process_eval=args.post_process_eval,
        device=str(device),
        amp=args.amp,
        sync_bn=args.sync_bn,
        full_finetune=args.full_finetune,
        finetune_blocks=args.finetune_blocks,
        encoder_lr=args.encoder_lr,
        output_dir=args.output_dir,
        local_rank=local_rank,
        world_size=world_size,
        distributed=is_ddp,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    cfg_fields = inspect.signature(RetrainConfig).parameters
    cfg = RetrainConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_fields})
    for k, v in cfg_kwargs.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    retrainer = Retrainer(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        boundary_teacher=boundary_teacher,
        scale_teacher=scale_teacher,
    )
    if args.resume:
        if not Path(args.resume).is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        retrainer.load_checkpoint(args.resume)

    if is_main:
        logger.info("Starting retraining ...")
    history = retrainer.train()

    if is_main:
        best_epoch = min(
            (r for r in history if "val_abs_rel" in r),
            key=lambda r: r["val_abs_rel"],
            default=history[-1],
        )
        logger.info(
            f"Retraining done. Best val abs_rel = "
            f"{best_epoch.get('val_abs_rel', float('nan')):.4f} "
            f"at epoch {best_epoch['epoch']}"
        )

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
