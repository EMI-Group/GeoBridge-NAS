#!/usr/bin/env python3
"""Evaluate a depth checkpoint on the supplied dataset and split.

Weights load strictly. Use --output to save full-precision metrics and settings.
See README.md for NYU and KITTI commands matching the public release.
"""

import argparse
import inspect
import logging
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best_d1.pth / epoch_*.pth")
    p.add_argument("--output", type=Path, help="Write recomputed metrics and evaluation settings as JSON.")
    p.add_argument("--arch", "--genotype", dest="arch", default=None)
    p.add_argument("--arch_path", default=None)
    p.add_argument("--arch_idx", type=int, default=0)
    p.add_argument("--decoder_baseline", choices=["tree", "vmamba", "localmamba"], default="tree")
    p.add_argument("--dataset", choices=["nyu", "tartanair", "matterport3d", "ddad", "kitti"], default="ddad")
    p.add_argument("--data_root", default=None)
    p.add_argument("--gt_root", default=None)
    p.add_argument("--filenames_file_eval", default=None)
    p.add_argument("--intrinsics_file", default=None)
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument("--dinov2_ckpt", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--depth_scale", type=float, default=None)
    p.add_argument("--max_depth", type=float, default=None)
    p.add_argument("--min_depth_eval", type=float, default=1e-3)
    p.add_argument("--max_depth_eval", type=float, default=None)
    p.add_argument("--eigen_crop", action="store_true", default=False)
    p.add_argument("--garg_crop", action="store_true", default=False)
    p.add_argument("--post_process_eval", action="store_true", default=False)
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Cap eval-set size. Point --filenames_file_eval at the TRAIN "
                        "split with e.g. 2000 to measure the train/val gap (over- vs underfit).")
    p.add_argument("--per_bin", action="store_true", default=False,
                   help="Break metrics down by GT depth bin to localise error.")
    p.add_argument("--bin_edges", default="0.001,10,20,30,40,50,60,80",
                   help="Comma-separated depth bin edges for --per_bin.")
    p.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast during evaluation.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()






def reeval(args) -> None:
    from ddn.search_space.tree import DecoderTree
    from ddn.student.student_net import StudentNet
    from ddn.data.nyu import NYUDepthV2Dataset
    from ddn.data.matterport3d import Matterport3DDataset
    from ddn.data.tartanair import TartanAirDataset
    from ddn.data.ddad import DDADDataset
    from ddn.data.kitti import KITTIEigenDataset
    from ddn.train.retrain import Retrainer, RetrainConfig
    from torch.utils.data import DataLoader

    # Dataset-specific defaults mirroring train_depth.py.
    if args.dataset == "ddad":
        # PixelFormer DDAD protocol (matches ddad.zip arg file): full-frame
        # resize at 608x960, 200 m ceiling, NO garg/eigen eval crop, flip
        # post-process at eval.
        args.depth_scale = args.depth_scale or 256.0
        args.max_depth = args.max_depth or 200.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 608, 960
        args.garg_crop = False
        args.eigen_crop = False
        args.post_process_eval = True
    elif args.dataset == "kitti":
        # Standard Eigen-split protocol mirroring train_depth.py: kb-crop 352x1216,
        # 80 m ceiling, garg crop, flip post-process.
        args.depth_scale = args.depth_scale or 256.0
        args.max_depth = args.max_depth or 80.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 352, 1216
        args.garg_crop = True
        args.eigen_crop = False
        args.post_process_eval = True
    else:
        args.depth_scale = args.depth_scale or 1000.0
        args.max_depth = args.max_depth or 10.0

    if args.decoder_baseline != "tree":
        arch = None
    elif args.arch:
        if args.arch.startswith("traditional_fpn("):
            from ddn.search_space.traditional import TraditionalDecoderArch
            arch = TraditionalDecoderArch.from_summary(args.arch, channels=args.decoder_channels)
        else:
            arch = DecoderTree.from_summary(args.arch, channels=args.decoder_channels)
    elif args.arch_path:
        with open(args.arch_path, "rb") as f:
            arch = pickle.load(f)[args.arch_idx]
    else:
        raise ValueError("Evaluation needs --arch '<genotype>' or --arch_path <pkl>")

    student = StudentNet(
        decoder_tree=arch,
        trainable_bias=False,
        decoder_channels=args.decoder_channels,
        weight_path=args.dinov2_ckpt,
        pretrained_encoder=False,
        min_depth=args.min_depth_eval,
        max_depth=args.max_depth,
        decoder_baseline=args.decoder_baseline,
    )
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = student.load_state_dict(sd, strict=True)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:3]}…")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}…")

    if not args.filenames_file_eval or not args.data_root:
        raise ValueError("Evaluation needs --data_root and --filenames_file_eval")

    common = dict(
        root=args.data_root,
        filenames_file=args.filenames_file_eval,
        img_size=(args.img_h, args.img_w),
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        max_samples=args.max_eval_samples,
    )
    if args.dataset == "nyu":
        val_ds = NYUDepthV2Dataset(split="val", **common)
    elif args.dataset == "matterport3d":
        val_ds = Matterport3DDataset(sparse_ratio=0.0, require_intrinsics=False, **common)
    elif args.dataset == "ddad":
        val_ds = DDADDataset(gt_root=args.gt_root, intrinsics_file=args.intrinsics_file, split="val", sparse_ratio=0.0, **common)
    elif args.dataset == "kitti":
        val_ds = KITTIEigenDataset(gt_root=args.gt_root, split="val", sparse_ratio=0.0, **common)
    elif args.dataset == "tartanair":
        val_ds = TartanAirDataset(root=args.data_root, split="val", img_size=(args.img_h, args.img_w))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Val samples: {len(val_ds)}")

    cfg_kwargs = dict(
        n_epochs=1,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval if args.max_depth_eval is not None else args.max_depth,
        eigen_crop=args.eigen_crop,
        garg_crop=args.garg_crop,
        post_process_eval=args.post_process_eval,
        device=args.device,
        amp=args.amp,
        distributed=False,
    )
    cfg_fields = inspect.signature(RetrainConfig).parameters
    cfg = RetrainConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_fields})

    retrainer = Retrainer(student=student, train_loader=val_loader, val_loader=val_loader, config=cfg)
    # Re-load weights: Retrainer.__init__ does not touch them, but enable_retrain_mode
    # only flips requires_grad, so the loaded weights above are intact.
    metrics = retrainer._validate()
    logger.info("── Re-eval metrics (image-averaged, same protocol as training) ──")
    logger.info("  " + " | ".join(f"{k}={metrics.get(k, float('nan')):.4f}" for k in METRIC_KEYS))

    if args.output:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "checkpoint": Path(args.ckpt).name,
            "architecture": arch.summary() if arch is not None else args.decoder_baseline,
            "dataset": args.dataset, "samples": len(val_ds), "metrics": metrics,
            "protocol": {"image_size": [args.img_h, args.img_w],
                         "depth_scale": args.depth_scale, "max_depth": args.max_depth,
                         "min_depth_eval": cfg.min_depth_eval, "max_depth_eval": cfg.max_depth_eval,
                         "eigen_crop": args.eigen_crop, "garg_crop": args.garg_crop,
                         "post_process_eval": args.post_process_eval, "amp": args.amp}
        }, indent=2) + "\n")

    if args.per_bin:
        per_bin_eval(retrainer, val_loader, cfg,
                     edges=[float(e) for e in args.bin_edges.split(",")])


def per_bin_eval(retrainer, val_loader, cfg, edges) -> None:
    """Pixel-weighted d1/abs_rel/rmse broken down by GT depth bin.

    Uses the exact same eval crop + post-process as Retrainer._validate, so the
    only difference from the headline number is pixel- (not image-) averaging.
    """
    import torch.nn.functional as F

    device = retrainer.device
    n = len(edges) - 1
    cnt = torch.zeros(n, dtype=torch.float64, device=device)
    d1c = torch.zeros(n, dtype=torch.float64, device=device)
    absrel = torch.zeros(n, dtype=torch.float64, device=device)
    sqerr = torch.zeros(n, dtype=torch.float64, device=device)
    lo, hi = cfg.min_depth_eval, cfg.max_depth_eval

    retrainer.student.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            gt = batch["depth"].to(device)
            vm = batch.get("valid_mask")
            mask = (vm.to(device).bool() if vm is not None
                    else (torch.isfinite(gt) & (gt > lo) & (gt < hi)))
            mask = retrainer._apply_eval_crop(mask).squeeze(1)

            pred = retrainer.model(images).get("depth")
            if cfg.post_process_eval:
                flip = retrainer.model(torch.flip(images, dims=[3])).get("depth")
                if flip is not None:
                    pred = retrainer._post_process_depth(pred, flip)
            if pred.shape[2:] != gt.shape[2:]:
                pred = F.interpolate(pred, size=gt.shape[2:], mode="bilinear", align_corners=False)
            pred = pred.squeeze(1)
            gt = gt.squeeze(1)

            p = pred[mask].clamp(lo, hi)
            g = gt[mask].clamp(lo, hi)
            if p.numel() == 0:
                continue
            thresh = torch.max(p / g, g / p)
            for i in range(n):
                sub = (g >= edges[i]) & (g < edges[i + 1])
                if not sub.any():
                    continue
                cnt[i] += sub.sum()
                d1c[i] += (thresh[sub] < 1.25).sum()
                absrel[i] += ((p[sub] - g[sub]).abs() / g[sub]).sum()
                sqerr[i] += ((p[sub] - g[sub]) ** 2).sum()

    total = cnt.sum().clamp_min(1)
    logger.info("── Per-depth-bin (pixel-weighted) ──")
    logger.info(f"  {'range(m)':>12} | {'%pix':>6} | {'d1':>7} | {'abs_rel':>7} | {'rmse':>7}")
    for i in range(n):
        c = cnt[i].clamp_min(1)
        logger.info(
            f"  {f'{edges[i]:.0f}-{edges[i+1]:.0f}':>12} | "
            f"{100*cnt[i]/total:6.2f} | "
            f"{(d1c[i]/c).item():7.4f} | "
            f"{(absrel[i]/c).item():7.4f} | "
            f"{(sqerr[i]/c).sqrt().item():7.4f}"
        )
    logger.info(
        f"  {'overall':>12} | {100.0:6.2f} | "
        f"{(d1c.sum()/total).item():7.4f} | "
        f"{(absrel.sum()/total).item():7.4f} | "
        f"{(sqerr.sum()/total).sqrt().item():7.4f}  (pixel-weighted)"
    )


def main():
    args = parse_args()
    reeval(args)


if __name__ == "__main__":
    main()
