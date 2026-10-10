#!/usr/bin/env python3
"""Benchmark DDN latency and throughput for decoder genotypes.

Examples:
    python scripts/benchmark_decoder.py \
        --gene 's32:M1D8FBSXNK3|s16:M1D8FBSXNK3|s8:M1D16FSXNK3|s4:M1D8FSXNK3'

    python scripts/benchmark_decoder.py \
        --gene_file genes.txt \
        --batch_sizes 1,4
"""

import argparse
import contextlib
import csv
import json
import pickle
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.search_space.tree import DecoderTree, make_fpn_template, random_valid_tree
from ddn.student.student_net import StudentNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure DDN latency and throughput for one or more decoder genotypes."
    )
    p.add_argument(
        "--gene",
        "--genes",
        "--arch",
        "--genotype",
        dest="genes",
        action="append",
        default=[],
        help="Direct decoder genotype string. Can be passed more than once.",
    )
    p.add_argument(
        "--gene_file",
        "--arch_file",
        dest="gene_file",
        default=None,
        help="Text/JSON file containing genotypes. Text mode reads one genotype per line.",
    )
    p.add_argument(
        "--arch_path",
        default=None,
        help="Pickle file from search, usually search_results/pareto_front.pkl.",
    )
    p.add_argument(
        "--arch_indices",
        default="all",
        help="Comma-separated indices for --arch_path, or 'all'. Default: all.",
    )
    p.add_argument(
        "--random",
        type=int,
        default=0,
        help="Also benchmark N randomly sampled valid decoder genotypes.",
    )
    p.add_argument("--img_h", type=int, default=480, help="Input height. Default: 480.")
    p.add_argument("--img_w", type=int, default=640, help="Input width. Default: 640.")
    p.add_argument(
        "--batch_sizes",
        default="1",
        help="Comma-separated batch sizes for throughput measurement. Default: 1.",
    )
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations per model.")
    p.add_argument("--iters", type=int, default=100, help="Timed iterations per repeat.")
    p.add_argument("--repeats", type=int, default=3, help="Timing repeats per model.")
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument(
        "--scope",
        choices=["full", "decoder"],
        default="full",
        help="'full' benchmarks encoder + decoder; 'decoder' benchmarks only the searched decoder.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional StudentNet checkpoint to load with strict=False.",
    )
    p.add_argument(
        "--dinov2_ckpt",
        default=None,
        help="Optional DINOv2 ViT-S checkpoint path for the encoder.",
    )
    p.add_argument(
        "--pretrained_encoder",
        action="store_true",
        help="Load pretrained DINOv2 encoder weights. Off by default to avoid download during benchmarking.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast fp16 while timing.")
    p.add_argument("--channels_last", action="store_true", help="Use channels-last memory format.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_json", default=None, help="Optional path to write JSON results.")
    p.add_argument("--save_csv", default=None, help="Optional path to write CSV results.")
    return p.parse_args()


def parse_int_list(text: str) -> Optional[List[int]]:
    if text.strip().lower() == "all":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_batch_sizes(text: str) -> List[int]:
    sizes = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch_sizes must contain positive integers")
    return sizes


def arch_from_any(obj: object, channels: int) -> DecoderTree:
    if isinstance(obj, DecoderTree):
        if obj.channels == channels:
            return obj
        return DecoderTree.from_summary(obj.summary(), channels=channels)
    if isinstance(obj, str):
        return DecoderTree.from_summary(obj, channels=channels)
    if isinstance(obj, dict):
        if obj.get("kind") == "tree_coded_mamba_trunk":
            arch = DecoderTree.from_dict(obj)
            if arch.channels != channels:
                arch = DecoderTree.from_summary(arch.summary(), channels=channels)
            return arch
        for key in ("arch", "genotype", "summary"):
            if key in obj:
                return DecoderTree.from_summary(str(obj[key]), channels=channels)
    raise TypeError(f"Cannot parse architecture from object of type {type(obj).__name__}")


def load_arch_file(path: Path, channels: int) -> List[DecoderTree]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for key in ("architectures", "archs", "pareto_front", "items"):
                if key in data:
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            raise ValueError(f"JSON architecture file must contain a list or dict: {path}")
        return [arch_from_any(item, channels) for item in data]

    archs = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        archs.append(DecoderTree.from_summary(text, channels=channels))
    return archs


def load_arch_path(path: Path, indices: Optional[List[int]], channels: int) -> List[DecoderTree]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, Sequence):
        raise ValueError(f"--arch_path must contain a sequence, got {type(data).__name__}")
    selected = range(len(data)) if indices is None else indices
    archs = []
    for idx in selected:
        if idx < 0 or idx >= len(data):
            raise IndexError(f"--arch_indices contains {idx}, but {path} has {len(data)} entries")
        archs.append(arch_from_any(data[idx], channels))
    return archs


def collect_arches(args: argparse.Namespace) -> List[Tuple[str, DecoderTree]]:
    archs: List[Tuple[str, DecoderTree]] = []

    for i, text in enumerate(args.genes):
        archs.append((f"gene_{len(archs)}", DecoderTree.from_summary(text, args.decoder_channels)))

    if args.gene_file:
        for arch in load_arch_file(Path(args.gene_file), args.decoder_channels):
            archs.append((f"gene_{len(archs)}", arch))

    if args.arch_path:
        indices = parse_int_list(args.arch_indices)
        for arch in load_arch_path(Path(args.arch_path), indices, args.decoder_channels):
            archs.append((f"gene_{len(archs)}", arch))

    for _ in range(args.random):
        archs.append((f"random_{len(archs)}", random_valid_tree(args.decoder_channels)))

    if not archs:
        archs.append(("template", make_fpn_template(args.decoder_channels)))
    return archs


def load_checkpoint(model: nn.Module, checkpoint: str) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")


def padded_token_hw(img_h: int, img_w: int, patch: int = 14) -> Tuple[int, int]:
    pad_h = (patch - img_h % patch) % patch
    pad_w = (patch - img_w % patch) % patch
    return (img_h + pad_h) // patch, (img_w + pad_w) // patch


def make_inputs(
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> object:
    if args.scope == "full":
        x = torch.randn(batch_size, 3, args.img_h, args.img_w, device=device)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    token_h, token_w = padded_token_hw(args.img_h, args.img_w)
    return [
        torch.randn(
            batch_size,
            args.decoder_channels,
            token_h,
            token_w,
            device=device,
        )
        for _ in range(4)
    ]


def forward_once(model: StudentNet, inputs: object, scope: str) -> object:
    if scope == "full":
        return model(inputs)  # type: ignore[arg-type]
    return model.decoder(inputs)  # type: ignore[arg-type]


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


def measure_once(
    model: StudentNet,
    inputs: object,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float]:
    with torch.inference_mode(), autocast_context(device, args.amp):
        for _ in range(args.warmup):
            forward_once(model, inputs, args.scope)
        sync_if_needed(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                forward_once(model, inputs, args.scope)
            end.record()
            sync_if_needed(device)
            total_ms = float(start.elapsed_time(end))
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        else:
            start_t = time.perf_counter()
            for _ in range(args.iters):
                forward_once(model, inputs, args.scope)
            total_ms = (time.perf_counter() - start_t) * 1000.0
            peak_mb = 0.0
    return total_ms / args.iters, peak_mb


def benchmark_arch(
    name: str,
    arch: DecoderTree,
    batch_sizes: Iterable[int],
    args: argparse.Namespace,
    device: torch.device,
) -> List[dict]:
    load_pretrained = bool(args.pretrained_encoder or args.dinov2_ckpt)
    model = StudentNet(
        decoder_tree=arch,
        trainable_bias=False,
        decoder_channels=args.decoder_channels,
        weight_path=args.dinov2_ckpt,
        pretrained_encoder=load_pretrained,
    ).to(device).eval()

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        model.to(device).eval()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    results = []
    for batch_size in batch_sizes:
        inputs = make_inputs(batch_size, args, device)
        latencies = []
        peaks = []
        for _ in range(args.repeats):
            latency_ms, peak_mb = measure_once(model, inputs, args, device)
            latencies.append(latency_ms)
            peaks.append(peak_mb)
        mean_ms = statistics.mean(latencies)
        std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        results.append(
            {
                "name": name,
                "batch_size": batch_size,
                "scope": args.scope,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "latency_ms_per_batch": mean_ms,
                "latency_std_ms": std_ms,
                "latency_ms_per_image": mean_ms / batch_size,
                "throughput_img_s": 1000.0 * batch_size / mean_ms,
                "peak_mem_mb": max(peaks),
                "decoder_flops": float(arch.flops(args.img_h, args.img_w)),
                "decoder_params": float(arch.params()),
                "genotype": arch.summary(),
            }
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def print_results(rows: List[dict]) -> None:
    header = (
        "name",
        "bs",
        "scope",
        "lat_ms/batch",
        "lat_ms/img",
        "img/s",
        "mem_mb",
        "dec_params",
    )
    print(
        f"{header[0]:<12} {header[1]:>4} {header[2]:>8} "
        f"{header[3]:>13} {header[4]:>11} {header[5]:>10} "
        f"{header[6]:>10} {header[7]:>11}"
    )
    for row in rows:
        print(
            f"{row['name']:<12} {row['batch_size']:>4} {row['scope']:>8} "
            f"{row['latency_ms_per_batch']:>13.3f} "
            f"{row['latency_ms_per_image']:>11.3f} "
            f"{row['throughput_img_s']:>10.2f} "
            f"{row['peak_mem_mb']:>10.1f} "
            f"{row['decoder_params']:>11.0f}"
        )


def write_outputs(rows: List[dict], args: argparse.Namespace) -> None:
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(rows, indent=2))
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.save_csv).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--iters and --repeats must be positive; --warmup must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    archs = collect_arches(args)

    rows: List[dict] = []
    print(
        f"Benchmarking {len(archs)} architecture(s) on {device}; "
        f"resolution={args.img_w}x{args.img_h}, batches={batch_sizes}, scope={args.scope}"
    )
    for name, arch in archs:
        rows.extend(benchmark_arch(name, arch, batch_sizes, args, device))

    print_results(rows)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
