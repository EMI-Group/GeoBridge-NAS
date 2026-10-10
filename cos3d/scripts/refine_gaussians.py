#!/usr/bin/env python3
"""Run vanilla 3DGS RGB/SSIM optimization from an externally supplied PLY.

The upstream Gaussian Splatting trainer normally initializes from COLMAP.
This wrapper changes only that initialization step: cameras, train/test split,
losses, optimizer, renderer and schedules remain the upstream implementation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _bootstrap_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--initial-ply", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    bootstrap = _bootstrap_arguments()
    gs_root = bootstrap.gs_root.resolve()
    initial_ply = bootstrap.initial_ply.resolve()
    if not (gs_root / "train.py").is_file():
        raise FileNotFoundError(f"Gaussian Splatting train.py not found: {gs_root}")
    if not initial_ply.is_file():
        raise FileNotFoundError(f"Initial Gaussian PLY not found: {initial_ply}")

    os.chdir(gs_root)
    sys.path.insert(0, str(gs_root))
    gs_train = importlib.import_module("train")
    original_scene = gs_train.Scene
    runtime = {"eval_only": False}

    class InitialPlyScene:
        def __new__(cls, args, gaussians, *scene_args, **scene_kwargs):
            scene = original_scene(
                args, gaussians, *scene_args, **scene_kwargs
            )
            gaussians.load_ply(
                str(initial_ply),
                getattr(args, "train_test_exp", False),
            )
            gaussians.spatial_lr_scale = scene.cameras_extent
            # load_ply replaces the COLMAP-initialized tensors, so the
            # screen-space cache must be resized before official densification
            # can use it.  DepthSplat exports DC color only; restart SH at
            # degree zero to preserve the upstream 1K/2K/3K activation schedule.
            gaussians.max_radii2D = gs_train.torch.zeros(
                gaussians.get_xyz.shape[0],
                device="cuda",
            )
            if not runtime["eval_only"]:
                gaussians.active_sh_degree = 0
            print(
                f"[initial-ply] loaded {gaussians.get_xyz.shape[0]} "
                f"Gaussians from {initial_ply}",
                flush=True,
            )
            return scene

    gs_train.Scene = InitialPlyScene

    parser = argparse.ArgumentParser(
        description="Optimize a supplied A3 Gaussian model with vanilla 3DGS"
    )
    lp = gs_train.ModelParams(parser)
    op = gs_train.OptimizationParams(parser)
    pp = gs_train.PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[1, 500, 1000, 2000],
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--initial-ply", type=Path, required=True)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Render the official test/train views once without optimization.",
    )
    args = parser.parse_args(sys.argv[1:])
    runtime["eval_only"] = args.eval_only
    args.save_iterations.append(args.iterations)

    gs_train.safe_state(args.quiet)
    if not args.disable_viewer:
        gs_train.network_gui.init(args.ip, args.port)
    gs_train.torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    if args.eval_only:
        print(f"Evaluating {args.model_path}")
        Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
        gaussians = gs_train.GaussianModel(
            dataset.sh_degree, op.extract(args).optimizer_type
        )
        scene = gs_train.Scene(dataset, gaussians)
        background = gs_train.torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=gs_train.torch.float32,
            device="cuda",
        )
        zero = gs_train.torch.tensor(0.0, device="cuda")
        gs_train.training_report(
            None,
            0,
            zero,
            zero,
            gs_train.l1_loss,
            0.0,
            [0],
            scene,
            gs_train.render,
            (
                pipeline,
                background,
                1.0,
                gs_train.SPARSE_ADAM_AVAILABLE,
                None,
                dataset.train_test_exp,
            ),
            dataset.train_test_exp,
        )
        print("\nEvaluation complete.")
        return

    print(f"Optimizing {args.model_path}")
    gs_train.training(
        dataset,
        op.extract(args),
        pipeline,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
