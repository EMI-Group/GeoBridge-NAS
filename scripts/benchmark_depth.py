#!/usr/bin/env python3
"""Benchmark latency for the fixed A1/A2/A3 DDN decoder genotypes.

Examples:
    python scripts/benchmark_depth.py --resolution nyu --batch_sizes 1,4
    python scripts/benchmark_depth.py --resolution kitti --scope decoder
    python scripts/benchmark_depth.py --resolution all --amp --save_csv reports/a123_latency.csv
"""

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.search_space.tree import DecoderTree
from ddn.student.student_net import StudentNet


ARCHES: Dict[str, str] = {
    "A3": "s32:((M2D8HWGXOK5>M1D16HWGXOK3)>R(M2D8FBSIOK3))|s16:(M2D8FBGXNK3+R(M1D16HWGIOK3))|s8:R(M1D8FSXNK3)|s4:M1D16FGINK3",
    "A2": "s32:M1D16FBGINK5|s16:M1D16FBSIOK3|s8:M1D8FGIOK3|s4:M1D8FBSIOK3",
    "A1": "s32:M1D16FBGXNK5|s16:M1D16FBSINK3|s8:M1D8FGIOK3|s4:R(M1D16FGINK3)",
}

RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "nyu": (480, 640),
    "kitti": (352, 1216),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark A1/A2/A3 latency.")
    p.add_argument("--resolution", choices=["nyu", "kitti", "all", "custom"], default="all")
    p.add_argument("--img_h", type=int, default=480, help="Used only with --resolution custom.")
    p.add_argument("--img_w", type=int, default=640, help="Used only with --resolution custom.")
    p.add_argument("--batch_sizes", default="1", help="Comma-separated batch sizes.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument("--scope", choices=["full", "decoder"], default="full")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast.")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_json", default=None)
    p.add_argument("--save_csv", default=None)
    return p.parse_args()


def parse_batch_sizes(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("--batch_sizes must contain positive integers")
    return values


def selected_resolutions(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    if args.resolution == "all":
        return [(name, hw[0], hw[1]) for name, hw in RESOLUTIONS.items()]
    if args.resolution == "custom":
        return [("custom", args.img_h, args.img_w)]
    h, w = RESOLUTIONS[args.resolution]
    return [(args.resolution, h, w)]


def padded_token_hw(img_h: int, img_w: int, patch: int = 14) -> Tuple[int, int]:
    pad_h = (patch - img_h % patch) % patch
    pad_w = (patch - img_w % patch) % patch
    return (img_h + pad_h) // patch, (img_w + pad_w) // patch


def make_inputs(
    batch_size: int,
    scope: str,
    img_h: int,
    img_w: int,
    channels: int,
    device: torch.device,
    channels_last: bool,
) -> object:
    if scope == "full":
        x = torch.randn(batch_size, 3, img_h, img_w, device=device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    token_h, token_w = padded_token_hw(img_h, img_w)
    return [
        torch.randn(batch_size, channels, token_h, token_w, device=device)
        for _ in range(4)
    ]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


def forward_once(model: nn.Module, inputs: object, scope: str):
    if scope == "full":
        return model(inputs)  # type: ignore[arg-type]
    return model(inputs)  # decoder receives the feature-list directly


def measure(
    model: nn.Module,
    inputs: object,
    scope: str,
    device: torch.device,
    amp: bool,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    with torch.inference_mode(), autocast(device, amp):
        for _ in range(warmup):
            forward_once(model, inputs, scope)
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                forward_once(model, inputs, scope)
            end.record()
            sync(device)
            total_ms = float(start.elapsed_time(end))
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                forward_once(model, inputs, scope)
            total_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = 0.0
    return total_ms / float(iters), peak_mb


def make_model(arch: DecoderTree, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.scope == "decoder":
        model = arch.materialize()
    else:
        model = StudentNet(
            decoder_tree=arch,
            trainable_bias=False,
            decoder_channels=args.decoder_channels,
            pretrained_encoder=False,
        )
    model = model.to(device).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def benchmark_one(
    arch_name: str,
    arch: DecoderTree,
    res_name: str,
    img_h: int,
    img_w: int,
    batch_sizes: Iterable[int],
    args: argparse.Namespace,
    device: torch.device,
) -> List[dict]:
    model = make_model(arch, args, device)
    rows: List[dict] = []
    for batch_size in batch_sizes:
        inputs = make_inputs(
            batch_size,
            args.scope,
            img_h,
            img_w,
            args.decoder_channels,
            device,
            args.channels_last,
        )
        latencies = []
        peaks = []
        for _ in range(args.repeats):
            latency_ms, peak_mb = measure(
                model,
                inputs,
                args.scope,
                device,
                args.amp,
                args.warmup,
                args.iters,
            )
            latencies.append(latency_ms)
            peaks.append(peak_mb)
        mean_ms = statistics.mean(latencies)
        std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        rows.append(
            {
                "arch": arch_name,
                "resolution": res_name,
                "img_h": img_h,
                "img_w": img_w,
                "scope": args.scope,
                "batch_size": batch_size,
                "latency_ms_per_batch": mean_ms,
                "latency_std_ms": std_ms,
                "latency_ms_per_image": mean_ms / batch_size,
                "throughput_img_s": 1000.0 * batch_size / mean_ms,
                "peak_mem_mb": max(peaks),
                "decoder_flops": arch.flops(img_h, img_w),
                "decoder_params": arch.params(),
                "genotype": arch.summary(),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def print_rows(rows: List[dict]) -> None:
    print(
        f"{'arch':<4} {'res':<6} {'size':>9} {'bs':>3} {'scope':>7} "
        f"{'ms/batch':>10} {'ms/img':>9} {'img/s':>9} {'memMB':>9}"
    )
    for r in rows:
        size = f"{r['img_h']}x{r['img_w']}"
        print(
            f"{r['arch']:<4} {r['resolution']:<6} {size:>9} "
            f"{r['batch_size']:>3} {r['scope']:>7} "
            f"{r['latency_ms_per_batch']:>10.3f} "
            f"{r['latency_ms_per_image']:>9.3f} "
            f"{r['throughput_img_s']:>9.2f} "
            f"{r['peak_mem_mb']:>9.1f}"
        )


def write_outputs(rows: List[dict], args: argparse.Namespace) -> None:
    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))
    if args.save_csv:
        path = Path(args.save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.iters <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("--iters and --repeats must be positive; --warmup must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    resolutions = selected_resolutions(args)
    archs = {
        name: DecoderTree.from_summary(genotype, channels=args.decoder_channels)
        for name, genotype in ARCHES.items()
    }

    rows: List[dict] = []
    for res_name, img_h, img_w in resolutions:
        for arch_name, arch in archs.items():
            rows.extend(
                benchmark_one(
                    arch_name,
                    arch,
                    res_name,
                    img_h,
                    img_w,
                    batch_sizes,
                    args,
                    device,
                )
            )

    print_rows(rows)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
