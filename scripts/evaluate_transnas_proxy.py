#!/usr/bin/env python3
"""Evaluate gradient-stability proxy on TransNAS-Bench-101.

The proxy is intentionally zero-cost: instantiate each architecture with random
initialization, run one synthetic classification backward pass, and use
log10(gradient sensitivity) as a stability objective.  Lower grad_obj is better.
The script compares this proxy ranking against tabular TransNAS-Bench-101
validation/test metrics.
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRANSNAS_ROOT = ROOT / "third_party" / "TransNASBench"
TRANSNAS_API_DIR = TRANSNAS_ROOT / "api"
TRANSNAS_LIB_DIR = TRANSNAS_ROOT / "lib"
for p in (ROOT, TRANSNAS_API_DIR, TRANSNAS_LIB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The official model files import torchsummary only for their __main__ demos.
if "torchsummary" not in sys.modules:
    torchsummary_stub = types.ModuleType("torchsummary")
    torchsummary_stub.summary = lambda *args, **kwargs: None
    sys.modules["torchsummary"] = torchsummary_stub
if "models.utils" not in sys.modules:
    models_utils_stub = types.ModuleType("models.utils")
    models_utils_stub.merge_list = lambda xs: [item for sub in xs for item in sub]
    sys.modules["models.utils"] = models_utils_stub

from api import TransNASBenchAPI  # noqa: E402
from models.net_infer.net_macro import MacroNet  # noqa: E402
from ddn.proxy.grad_stability import grad_stability_objective  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TASK_METRICS = {
    "class_scene": "valid_top1",
    "class_object": "valid_top1",
    "room_layout": "valid_neg_loss",
    "jigsaw": "valid_top1",
    "segmentsemantic": "valid_mIoU",
    "normal": "valid_ssim",
    "autoencoder": "valid_ssim",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_path", default="data/transnasbench101/transnas-bench_v10141024.pth")
    p.add_argument("--output_dir", default="transnas101_grad_proxy")
    p.add_argument("--search_space", choices=["macro", "micro", "all"], default="micro")
    p.add_argument("--tasks", default="all", help="Comma list or 'all'.")
    p.add_argument("--metric_mode", choices=["best", "final"], default="best")
    p.add_argument("--max_arches", type=int, default=256, help="0 means all architectures in the chosen search space.")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--proxy_seed", type=int, default=123)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_values(values: List[float], higher_better: bool = True) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    xv = sum((v - xm) ** 2 for v in x)
    yv = sum((v - ym) ** 2 for v in y)
    if xv <= 0 or yv <= 0:
        return float("nan")
    cov = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    return cov / math.sqrt(xv * yv)


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_values(x, higher_better=True), rank_values(y, higher_better=True))


def kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (x[i] - x[j]) * (y[i] - y[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")


def hit_at_k(proxy_scores: List[float], real_scores: List[float], k: int, real_top_frac: float = 0.1) -> float:
    if not proxy_scores:
        return float("nan")
    k = min(k, len(proxy_scores))
    n_real_top = max(1, int(math.ceil(len(real_scores) * real_top_frac)))
    proxy_top = set(sorted(range(len(proxy_scores)), key=lambda i: proxy_scores[i], reverse=True)[:k])
    real_top = set(sorted(range(len(real_scores)), key=lambda i: real_scores[i], reverse=True)[:n_real_top])
    return len(proxy_top & real_top) / max(1, min(k, n_real_top))


def ndcg_at_k(proxy_scores: List[float], real_scores: List[float], k: int) -> float:
    if not proxy_scores:
        return float("nan")
    k = min(k, len(proxy_scores))
    order = sorted(range(len(proxy_scores)), key=lambda i: proxy_scores[i], reverse=True)[:k]
    ideal = sorted(range(len(real_scores)), key=lambda i: real_scores[i], reverse=True)[:k]
    min_score = min(real_scores)
    shifted = [s - min_score for s in real_scores]

    def dcg(indices: Iterable[int]) -> float:
        return sum((2.0 ** shifted[idx] - 1.0) / math.log2(rank + 2.0) for rank, idx in enumerate(indices))

    ideal_dcg = dcg(ideal)
    return dcg(order) / ideal_dcg if ideal_dcg > 0 else float("nan")


def choose_arches(api: TransNASBenchAPI, args) -> List[str]:
    spaces = api.search_spaces if args.search_space == "all" else [args.search_space]
    arches: List[str] = []
    for space in spaces:
        arches.extend(api.get_arch_list(space))
    arches = sorted(set(arches))
    if args.max_arches and args.max_arches > 0 and args.max_arches < len(arches):
        rng = random.Random(args.sample_seed)
        arches = rng.sample(arches, args.max_arches)
    return arches


def choose_tasks(api: TransNASBenchAPI, args) -> List[str]:
    if args.tasks == "all":
        return list(api.task_list)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in api.task_list]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Available: {api.task_list}")
    return tasks


def compute_grad_proxy(arch: str, args, arch_idx: int) -> Dict[str, float]:
    set_seed(args.proxy_seed + arch_idx)
    device = torch.device(args.device)
    model = MacroNet(
        arch,
        structure="full",
        input_dim=(args.image_size, args.image_size),
        num_classes=args.num_classes,
    ).to(device)
    model.train()

    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    sq_sum = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq_sum += float(torch.sum(g * g).item())
        n_params += g.numel()
    grad_l2 = math.sqrt(max(sq_sum, 0.0))
    grad_rms = grad_l2 / math.sqrt(max(n_params, 1))
    del model, x, y, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "grad_l2": grad_l2,
        "grad_rms": grad_rms,
        "grad_obj": grad_stability_objective(grad_rms),
    }


def query_real_metrics(api: TransNASBenchAPI, arch: str, tasks: List[str], mode: str) -> Dict[str, float]:
    result = {}
    for task in tasks:
        metric = DEFAULT_TASK_METRICS[task]
        value = api.get_single_metric(arch, task, metric, mode=mode)
        result[f"{task}:{metric}"] = float(value)
    return result


def summarize(rows: List[Dict], tasks: List[str], mode: str) -> Dict:
    proxy_small_grad = [float(r["proxy_score"]) for r in rows]
    proxy_large_grad = [-float(r["proxy_score"]) for r in rows]
    summary = {"n": len(rows), "metric_mode": mode, "tasks": {}}
    for task in tasks:
        metric = DEFAULT_TASK_METRICS[task]
        key = f"{task}:{metric}"
        real = [float(r[key]) for r in rows]
        k10 = max(1, int(math.ceil(0.1 * len(rows))))
        summary["tasks"][key] = {
            "spearman_small_grad": spearman(proxy_small_grad, real),
            "kendall_small_grad": kendall_tau(proxy_small_grad, real),
            "hit_at_10pct_small_grad": hit_at_k(proxy_small_grad, real, k=k10, real_top_frac=0.1),
            "ndcg_at_10pct_small_grad": ndcg_at_k(proxy_small_grad, real, k=k10),
            "spearman_large_grad": spearman(proxy_large_grad, real),
            "kendall_large_grad": kendall_tau(proxy_large_grad, real),
            "hit_at_10pct_large_grad": hit_at_k(proxy_large_grad, real, k=k10, real_top_frac=0.1),
            "ndcg_at_10pct_large_grad": ndcg_at_k(proxy_large_grad, real, k=k10),
            "real_best": max(real),
            "real_mean": sum(real) / len(real),
        }
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = TransNASBenchAPI(args.bench_path)
    tasks = choose_tasks(api, args)
    arches = choose_arches(api, args)
    logger.info(
        "Loaded TransNAS-Bench-101: spaces=%s tasks=%s selected_arches=%d device=%s",
        args.search_space,
        tasks,
        len(arches),
        args.device,
    )

    rows = []
    for i, arch in enumerate(arches):
        proxy = compute_grad_proxy(arch, args, i)
        metrics = query_real_metrics(api, arch, tasks, args.metric_mode)
        row = {"arch_idx": api.arch2index(arch), "arch": arch, **proxy, **metrics}
        rows.append(row)
        if (i + 1) == 1 or (args.log_every > 0 and (i + 1) % args.log_every == 0):
            logger.info(
                "[%d/%d] grad_rms=%.4e grad_obj=%.4f arch=%s",
                i + 1,
                len(arches),
                row["grad_rms"],
                row["grad_obj"],
                arch,
            )

    ranks = rank_values([r["grad_obj"] for r in rows], higher_better=False)
    n = max(len(rows), 1)
    for row, rank in zip(rows, ranks):
        row["proxy_rank_grad"] = rank
        row["proxy_score"] = (n + 1 - rank) / n
    rows.sort(key=lambda r: r["proxy_rank_grad"])

    summary = summarize(rows, tasks, args.metric_mode)
    with open(out_dir / "transnas101_grad_proxy.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    fieldnames = [
        "proxy_rank_grad",
        "proxy_score",
        "arch_idx",
        "arch",
        "grad_l2",
        "grad_rms",
        "grad_obj",
    ]
    for task in tasks:
        fieldnames.append(f"{task}:{DEFAULT_TASK_METRICS[task]}")
    with open(out_dir / "transnas101_grad_proxy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
