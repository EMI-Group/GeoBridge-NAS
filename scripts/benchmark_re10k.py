#!/usr/bin/env python3
"""Benchmark the actual DDN-DepthSplat RealEstate10K inference path.

The benchmark uses real Re10K evaluation batches and trained full-pipeline
checkpoints.  It reports the cost of (i) encoding two context views into
Gaussians, (ii) rendering one target view, (iii) end-to-end inference for one
target view, and (iv) the native evaluation sample with all target views.

Timing deliberately excludes data loading.  CUDA events bracket 100 forwards
at a time after warm-up, which measures steady-state batch-1 latency without
mixing in CPU decoding or metric computation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
DEPTHSPLAT_ROOT = ROOT / "third_party" / "depthsplat"
for path in (ROOT, DEPTHSPLAT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config import load_typed_root_config  # noqa: E402
from src.dataset import get_dataset  # noqa: E402
from src.global_cfg import set_cfg  # noqa: E402
from src.model.decoder import get_decoder  # noqa: E402
from src.model.encoder import get_encoder  # noqa: E402


ARCHITECTURES = {
    "DDN-A1": (
        "s32:M1D16FBGXNK5|s16:M1D16FBSINK3|"
        "s8:M1D8FGIOK3|s4:R(M1D16FGINK3)"
    ),
    "DDN-A2": (
        "s32:M1D16FBGINK5|s16:M1D16FBSIOK3|"
        "s8:M1D8FGIOK3|s4:M1D8FBSIOK3"
    ),
    "DDN-A3": (
        "s32:((M2D8HWGXOK5>M1D16HWGXOK3)>R(M2D8FBSIOK3))|"
        "s16:(M2D8FBGXNK3+R(M1D16HWGIOK3))|"
        "s8:R(M1D8FSXNK3)|s4:M1D16FGINK3"
    ),
}


class Re10KPipeline(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, batch: dict[str, Any]):
        gaussians = self.encoder(
            batch["context"],
            global_step=0,
            deterministic=False,
        )
        if isinstance(gaussians, dict):
            gaussians = gaussians["gaussians"]
        return gaussians

    def render(self, gaussians: Any, batch: dict[str, Any], one_target: bool):
        target = batch["target"]
        sl = slice(0, 1) if one_target else slice(None)
        h, w = target["image"].shape[-2:]
        return self.decoder.forward(
            gaussians,
            target["extrinsics"][:, sl],
            target["intrinsics"][:, sl],
            target["near"][:, sl],
            target["far"][:, sl],
            (h, w),
            depth_mode=None,
        )

    def forward(self, batch: dict[str, Any], one_target: bool = True):
        return self.render(self.encode(batch), batch, one_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="NAME=PATH, where NAME is DDN-A1, DDN-A2, or DDN-A3",
    )
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--evaluation-index",
        default=str(DEPTHSPLAT_ROOT / "assets" / "evaluation_index_re10k.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "latency_re10k"),
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp_fp16"),
        default="fp32",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_checkpoints(items: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if name not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[name] = path
    missing = set(ARCHITECTURES) - set(checkpoints)
    if missing:
        raise ValueError(f"Missing checkpoints for {sorted(missing)}")
    return checkpoints


def compose_config(
    architecture: str,
    data_root: Path,
    evaluation_index: Path,
):
    overrides = [
        "+experiment=re10k",
        "mode=test",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.num_context_views=2",
        "dataset.image_shape=[256,256]",
        "dataset.augment=false",
        "dataset.skip_bad_shape=false",
        "dataset.test_chunk_interval=10",
        "data_loader.test.batch_size=1",
        "data_loader.test.num_workers=0",
        "data_loader.test.persistent_workers=false",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=2",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.monodepth_vit_type=vits",
        "model.encoder.student_mono=true",
        "model.encoder.student_mono_freeze=true",
        "model.encoder.student_mono_pretrained_encoder=false",
        "wandb.mode=disabled",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(DEPTHSPLAT_ROOT / "config"),
    ):
        cfg_dict = compose(config_name="main", overrides=overrides)
    cfg_dict.dataset.roots = [str(data_root)]
    cfg_dict.dataset.view_sampler.index_path = str(evaluation_index)
    cfg_dict.model.encoder.student_mono_arch = architecture
    # The full pipeline checkpoint below contains the trained StudentMonoShim.
    cfg_dict.model.encoder.student_mono_ckpt = None
    set_cfg(cfg_dict)
    return cfg_dict, load_typed_root_config(cfg_dict)


def extract_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu")
    state = loaded
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(loaded, dict) and key in loaded:
            state = loaded[key]
            break
    if not isinstance(state, dict):
        raise TypeError(f"No state dictionary in {checkpoint}")
    cleaned = {}
    for name, value in state.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def load_matching_state(
    model: nn.Module,
    checkpoint: Path,
) -> dict[str, int]:
    source = extract_state(checkpoint)
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    shape_mismatch = sum(
        key in target and tuple(value.shape) != tuple(target[key].shape)
        for key, value in source.items()
    )
    missing, unexpected = model.load_state_dict(matched, strict=False)
    student_target = [key for key in target if ".pretrained.student." in key]
    student_matched = [key for key in matched if ".pretrained.student." in key]
    if len(student_matched) != len(student_target):
        raise RuntimeError(
            f"Student checkpoint mismatch for {checkpoint}: "
            f"matched {len(student_matched)}/{len(student_target)} tensors"
        )
    if len(matched) != len(target) or shape_mismatch:
        raise RuntimeError(
            f"Incomplete checkpoint tensor match for {checkpoint}: "
            f"{len(matched)}/{len(target)}"
        )
    return {
        "checkpoint_tensors": len(source),
        "model_tensors": len(target),
        "matched_tensors": len(matched),
        "student_tensors": len(student_target),
        "shape_mismatch": shape_mismatch,
        "missing_after_filter": len(missing),
        "unexpected_after_filter": len(unexpected),
    }


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def load_real_batches(
    cfg: Any,
    encoder: nn.Module,
    sample_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = get_dataset(cfg.dataset, "test", None)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    shim = encoder.get_data_shim()
    batches = []
    for batch in loader:
        batch = shim(batch)
        batch = move_to_device(batch, device)
        if batch["context"]["image"].shape[1] != 2:
            continue
        batches.append(batch)
        if len(batches) == sample_count:
            break
    if len(batches) != sample_count:
        raise RuntimeError(f"Loaded only {len(batches)}/{sample_count} batches")
    return batches


def precision_context(device: torch.device, precision: str):
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=precision == "amp_fp16",
        )
    return contextlib.nullcontext()


def measure(
    fn: Callable[[int], Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    repeat_ms = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), precision_context(device, args.precision):
        for repeat in range(args.repeats):
            for index in range(args.warmup):
                fn(index)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for index in range(args.iters):
                fn(index)
            end.record()
            torch.cuda.synchronize(device)
            repeat_ms.append(float(start.elapsed_time(end)) / args.iters)
    return {
        "mean_ms": statistics.mean(repeat_ms),
        "std_ms": statistics.pstdev(repeat_ms),
        "min_ms": min(repeat_ms),
        "max_ms": max(repeat_ms),
        "repeat_ms": repeat_ms,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def environment() -> dict[str, Any]:
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()
    except Exception as exc:  # pragma: no cover - diagnostics only
        smi = [repr(exc)]
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": smi,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (output_dir / "summary.csv").open("w", newline="") as file:
            flat_rows = [{**row, "repeat_ms": json.dumps(row["repeat_ms"])} for row in rows]
            writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("samples/iters/repeats must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoints = parse_checkpoints(args.checkpoint)
    data_root = Path(args.data_root).expanduser().resolve()
    evaluation_index = Path(args.evaluation_index).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not evaluation_index.is_file():
        raise FileNotFoundError(evaluation_index)

    os.chdir(DEPTHSPLAT_ROOT)
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(json.dumps(environment(), indent=2))
    protocol = {
        "dataset": "RealEstate10K evaluation split",
        "image_shape": [256, 256],
        "batch_size": 1,
        "context_views": 2,
        "data_loading_included": False,
        "precision": args.precision,
        "warmup_per_repeat": args.warmup,
        "timed_iterations_per_repeat": args.iters,
        "repeats": args.repeats,
        "real_batches": args.samples,
        "timing": "CUDA events around a continuous forward loop; synchronized at boundaries",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows: list[dict[str, Any]] = []
    for architecture_name, architecture in ARCHITECTURES.items():
        print(f"Building {architecture_name}", flush=True)
        _, cfg = compose_config(architecture, data_root, evaluation_index)
        encoder, _ = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder, cfg.dataset)
        model = Re10KPipeline(encoder, decoder)
        load_info = load_matching_state(model, checkpoints[architecture_name])
        model.to(device).eval().requires_grad_(False)

        batches = load_real_batches(cfg, model.encoder, args.samples, device)
        target_counts = [int(batch["target"]["image"].shape[1]) for batch in batches]
        if len(set(target_counts)) != 1:
            raise RuntimeError(f"Variable target counts are unsupported: {target_counts}")
        native_targets = target_counts[0]

        with torch.inference_mode(), precision_context(device, args.precision):
            cached_gaussians = [model.encode(batch) for batch in batches]
        torch.cuda.synchronize(device)

        def select(index: int):
            return index % len(batches)

        tasks: list[tuple[str, Callable[[int], Any], int]] = [
            (
                "encoder_2_context",
                lambda index: model.encode(batches[select(index)]),
                0,
            ),
            (
                "renderer_1_target",
                lambda index: model.render(
                    cached_gaussians[select(index)],
                    batches[select(index)],
                    True,
                ),
                1,
            ),
            (
                "end_to_end_1_target",
                lambda index: model(batches[select(index)], one_target=True),
                1,
            ),
            (
                f"end_to_end_{native_targets}_targets",
                lambda index: model(batches[select(index)], one_target=False),
                native_targets,
            ),
        ]

        for scope, fn, target_views in tasks:
            print(f"  measuring {scope}", flush=True)
            result = measure(fn, args, device)
            row = {
                "architecture": architecture_name,
                "scope": scope,
                "precision": args.precision,
                "batch_size": 1,
                "context_views": 2,
                "target_views": target_views,
                "height": 256,
                "width": 256,
                "samples": args.samples,
                "warmup": args.warmup,
                "iters": args.iters,
                "repeats": args.repeats,
                "checkpoint": str(checkpoints[architecture_name]),
                "genotype": architecture,
                **load_info,
                **result,
            }
            rows.append(row)
            write_results(rows, output_dir)
            print(
                f"    {result['mean_ms']:.3f} +/- {result['std_ms']:.3f} ms",
                flush=True,
            )

        del cached_gaussians, batches, model, encoder, decoder
        torch.cuda.empty_cache()

    write_results(rows, output_dir)
    (output_dir / "COMPLETED.txt").write_text(
        f"completed_rows={len(rows)}\n"
        f"architectures={len(ARCHITECTURES)}\n"
    )
    print(f"Completed {len(rows)} measurements", flush=True)


if __name__ == "__main__":
    main()
