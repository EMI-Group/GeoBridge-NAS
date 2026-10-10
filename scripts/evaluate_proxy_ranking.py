#!/usr/bin/env python3
"""Measure NAS proxy rank consistency on NYU Depth V2.

The script samples architectures from a search_log.json generation, trains
each sampled candidate with DDP on NYU v2, evaluates real depth metrics, and
reports rank-correlation / top-k hit metrics between proxy scores and downstream
validation performance.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddn.data.nyu import NYUDepthV2Dataset
from ddn.data.kitti import KITTIEigenDataset
from ddn.search_space.tree import DecoderTree
from ddn.student.student_net import StudentNet
from ddn.train.losses import SILogLoss
from ddn.utils.metrics import compute_depth_metrics
from ddn.proxy.grad_stability import grad_stability_objective


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--search_log", default="search_results/search_log.json")
    p.add_argument("--gen", type=int, default=-1, help="Generation index; -1 means last generation.")
    p.add_argument("--random_archs", type=int, default=0, help=(
        "ZCP-protocol mode: ignore search_log and train N architectures sampled "
        "uniformly at random from the search space (an unbiased, wide-quality pool, "
        "the way NAS-Bench proxy-correlation pools are built). 0 = use search_log. "
        "This mode exports trained candidates; proxy scores are not computed."))
    p.add_argument("--random_seed", type=int, default=12345,
                   help="Seed for the random architecture pool (reproducible).")
    p.add_argument("--arch_list_file", default="", help=(
        "Explicit genotype list, one per line; overrides --search_log and "
        "--random_archs. Used by the short-training study so that exactly the "
        "published matched pool is trained."))
    p.add_argument("--gen_pool", default="", help=(
        "Pool architectures across generations for a WIDE quality range (proper "
        "ZCP ranking validation). 'all' = every generation; or a comma list e.g. "
        "'0,15,29'. Empty = single generation given by --gen. With pooling, use "
        "--selection random to avoid proxy-stratification bias."))
    p.add_argument("--num_candidates", type=int, default=32)
    p.add_argument("--selection", choices=["uniform_proxy", "top_proxy", "random"], default="uniform_proxy")
    p.add_argument("--dataset", choices=["nyu", "kitti"], default="nyu")
    p.add_argument("--data_root", required=True)
    p.add_argument("--gt_root", default=None, help="Depth root for KITTI; defaults to data_root.")
    p.add_argument("--filenames_file", required=True)
    p.add_argument("--filenames_file_eval", required=True)
    p.add_argument("--output_dir", default="rank_consistency_nyu")
    p.add_argument("--dinov2_ckpt", default=None)
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--depth_scale", type=float, default=None)
    p.add_argument("--max_depth", type=float, default=None)
    p.add_argument("--do_kb_crop", action="store_true", default=False)
    p.add_argument("--do_random_rotate", action="store_true", default=False)
    p.add_argument("--degree", type=float, default=2.5)
    p.add_argument("--use_right", action="store_true", default=False)
    p.add_argument("--min_depth_eval", type=float, default=1e-3)
    p.add_argument("--max_depth_eval", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="If set, overrides --max_steps to exactly this many epochs "
                        "(max_steps = len(train_loader) * epochs), matching the paper's epoch budget.")
    p.add_argument("--max_steps", type=int, default=25000)
    p.add_argument("--val_every_steps", type=int, default=1000)
    p.add_argument("--save_every_steps", type=int, default=1000)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_encoder_bias", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--nan_patience", type=int, default=20,
                   help="Abort an architecture only after this many CONSECUTIVE non-finite "
                        "losses (true divergence). Isolated AMP/fp16 spikes are skipped and "
                        "training continues, preserving the best metrics already reached.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hit_k", type=int, default=5)
    p.add_argument("--real_top_frac", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def setup_ddp(args):
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    return local_rank, world_size, True


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def _rank_values(values: List[float], higher_better: bool = True) -> List[float]:
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


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rank_values(x, higher_better=True)
    ry = _rank_values(y, higher_better=True)
    return pearson(rx, ry)


def kendall_tau(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")


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


def ndcg_at_k(proxy_scores: List[float], real_scores: List[float], k: int) -> float:
    # min-max normalize relevance to [0,1] so the metric is well-defined for any
    # higher-is-better target (including negated error metrics like -abs_rel).
    lo, hi = min(real_scores), max(real_scores)
    rng = hi - lo
    rel = [(v - lo) / rng if rng > 0 else 0.0 for v in real_scores]
    order = sorted(range(len(proxy_scores)), key=lambda i: proxy_scores[i], reverse=True)[:k]
    ideal = sorted(range(len(rel)), key=lambda i: rel[i], reverse=True)[:k]

    def dcg(indices: Iterable[int]) -> float:
        return sum(rel[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(indices))

    ideal_dcg = dcg(ideal)
    return dcg(order) / ideal_dcg if ideal_dcg > 0 else float("nan")


def hit_at_k(proxy_scores: List[float], real_scores: List[float], k: int, real_top_frac: float) -> float:
    n_real_top = max(1, int(math.ceil(len(real_scores) * real_top_frac)))
    proxy_top = set(sorted(range(len(proxy_scores)), key=lambda i: proxy_scores[i], reverse=True)[:k])
    real_top = set(sorted(range(len(real_scores)), key=lambda i: real_scores[i], reverse=True)[:n_real_top])
    return len(proxy_top & real_top) / max(1, min(k, n_real_top))


def _select_candidates(candidates: List[Dict], args) -> List[Dict]:
    if args.num_candidates <= 0 or args.num_candidates >= len(candidates):
        return list(candidates)
    if args.selection == "top_proxy":
        return list(candidates[:args.num_candidates])
    if args.selection == "random":
        rng = random.Random(args.seed)
        sampled = rng.sample(candidates, args.num_candidates)
        return sorted(sampled, key=lambda c: c["proxy_score"], reverse=True)

    if args.num_candidates == 1:
        return [candidates[0]]
    last = len(candidates) - 1
    indices = sorted({round(i * last / (args.num_candidates - 1)) for i in range(args.num_candidates)})
    selected = [candidates[i] for i in indices]
    if len(selected) < args.num_candidates:
        used = set(indices)
        for i in range(len(candidates)):
            if i not in used:
                selected.append(candidates[i])
                used.add(i)
                if len(selected) == args.num_candidates:
                    break
    return sorted(selected, key=lambda c: c["proxy_score"], reverse=True)


def _random_arch_pool(args) -> List[Dict]:
    """ZCP-protocol pool: N unique architectures sampled uniformly at random from
    the search space, independent of any search trajectory. Proxy columns are left
    as NaN here; this path exports the candidate pool and training results."""
    from ddn.search_space.tree import random_valid_tree

    saved = random.getstate()
    random.seed(args.random_seed)
    seen: Dict[str, object] = {}
    tries, cap = 0, args.random_archs * 100
    while len(seen) < args.random_archs and tries < cap:
        tries += 1
        t = random_valid_tree(args.decoder_channels)
        if not t.validate():
            continue
        s = t.summary()
        seen.setdefault(s, t)
    random.setstate(saved)

    candidates = []
    for s, t in seen.items():
        try:
            flops_g = float(t.flops(args.img_h, args.img_w)) / 1e9
        except Exception:
            flops_g = float("nan")
        candidates.append({
            "arch": s, "grad": float("nan"), "grad_obj": float("nan"),
            "proxy_score": float("nan"), "proxy_rank_grad": 0,
            "flops_g": flops_g, "params_m": float(t.params()) / 1e6,
        })
    logger.info(
        "ZCP-protocol pool: %d unique random architectures (seed=%d, requested=%d).",
        len(candidates), args.random_seed, args.random_archs,
    )
    return candidates


def _arch_list_pool(args) -> List[Dict]:
    """Pool given explicitly as genotype strings, one per line.

    Used for the short-training study: it must train exactly the architectures
    of the published matched pool, so the short-horizon ranking can be compared
    against the same fully retrained d1 values.
    """
    from ddn.search_space.tree import DecoderTree

    candidates = []
    with open(args.arch_list_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            t = DecoderTree.from_summary(s, channels=args.decoder_channels)
            if not t.validate():
                raise ValueError(f"Invalid genotype in {args.arch_list_file}: {s}")
            try:
                flops_g = float(t.flops(args.img_h, args.img_w)) / 1e9
            except Exception:
                flops_g = float("nan")
            candidates.append({
                "arch": s, "grad": float("nan"), "grad_obj": float("nan"),
                "proxy_score": float("nan"), "proxy_rank_grad": 0,
                "flops_g": flops_g, "params_m": float(t.params()) / 1e6,
            })
    logger.info("Explicit arch pool: %d architectures from %s",
                len(candidates), args.arch_list_file)
    return candidates


def load_candidates(args) -> List[Dict]:
    if getattr(args, "arch_list_file", ""):
        return _arch_list_pool(args)
    if args.random_archs and args.random_archs > 0:
        return _random_arch_pool(args)

    with open(args.search_log) as f:
        logs = json.load(f)

    if args.gen_pool:
        if args.gen_pool.strip().lower() == "all":
            gen_idxs = list(range(len(logs)))
        else:
            gen_idxs = [int(g) for g in args.gen_pool.split(",") if g.strip() != ""]
        individuals = []
        for gi in gen_idxs:
            individuals.extend(logs[gi]["individuals"])
        gen_label = f"pool[{args.gen_pool}] ({len(gen_idxs)} gens)"
    else:
        gen_entry = logs[args.gen]
        individuals = gen_entry["individuals"]
        gen_label = str(gen_entry.get("gen", args.gen))

    unique = {}
    for item in individuals:
        arch = item["arch"]
        if arch not in unique:
            unique[arch] = dict(item)
    candidates = list(unique.values())
    if not candidates:
        raise ValueError(
            f"No candidates found in {args.search_log} (gen={args.gen}, pool={args.gen_pool!r})."
        )

    grad_objs = [
        float(c.get("grad_obj", grad_stability_objective(float(c["grad"]))))
        for c in candidates
    ]
    grad_ranks = _rank_values(grad_objs, higher_better=False)
    n = max(len(candidates), 1)
    for c, go, rg in zip(candidates, grad_objs, grad_ranks):
        c["grad_obj"] = go
        c["proxy_score"] = (n + 1 - rg) / n
        c["proxy_rank_grad"] = rg

    candidates = sorted(candidates, key=lambda c: c["proxy_score"], reverse=True)
    sampled = _select_candidates(candidates, args)
    logger.info(
        "Loaded gen=%s generation_candidates=%d selected=%d selection=%s",
        gen_label,
        len(candidates),
        len(sampled),
        args.selection,
    )
    return sampled


def build_loaders(args, is_ddp: bool):
    if args.dataset == "nyu":
        train_ds = NYUDepthV2Dataset(
            root=args.data_root,
            filenames_file=args.filenames_file,
            split="train",
            max_samples=args.max_train_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
        )
        val_ds = NYUDepthV2Dataset(
            root=args.data_root,
            filenames_file=args.filenames_file_eval,
            split="val",
            max_samples=args.max_val_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
        )
    elif args.dataset == "kitti":
        train_ds = KITTIEigenDataset(
            root=args.data_root,
            gt_root=args.gt_root,
            filenames_file=args.filenames_file,
            split="train",
            max_samples=args.max_train_samples,
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
            max_samples=args.max_val_samples,
            img_size=(args.img_h, args.img_w),
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            do_kb_crop=args.do_kb_crop,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _candidate_ckpt_path(args, arch_idx: int) -> Path:
    return Path(args.output_dir) / "candidate_checkpoints" / f"candidate_{arch_idx:03d}.pth"


def _save_candidate_checkpoint(
    args,
    arch_idx: int,
    model,
    optimizer,
    scaler,
    step: int,
    total_loss: float,
    best_metrics: Dict[str, float],
    best_d1: float,
    is_main: bool,
):
    if not is_main:
        return
    ckpt_dir = Path(args.output_dir) / "candidate_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _candidate_ckpt_path(args, arch_idx)
    payload = {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "total_loss": total_loss,
        "best_metrics": best_metrics,
        "best_d1": best_d1,
    }
    torch.save(payload, path)


def _load_candidate_checkpoint(args, arch_idx: int, model, optimizer, scaler, device: torch.device):
    path = _candidate_ckpt_path(args, arch_idx)
    if not args.resume or not path.exists():
        return 0, 0.0, {}, -float("inf")
    ckpt = torch.load(path, map_location=device)
    _unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt.get("step", 0)),
        float(ckpt.get("total_loss", 0.0)),
        dict(ckpt.get("best_metrics") or {}),
        float(ckpt.get("best_d1", -float("inf"))),
    )


def train_and_eval_arch(
    args,
    arch_text: str,
    arch_idx: int,
    train_loader,
    train_sampler,
    val_loader,
    is_ddp: bool,
    is_main: bool,
) -> Dict[str, float]:
    set_seed(args.seed + arch_idx)
    device = torch.device(args.device)
    arch = DecoderTree.from_summary(arch_text, channels=args.decoder_channels)
    model = StudentNet(
        decoder_tree=arch,
        trainable_bias=False,
        decoder_channels=args.decoder_channels,
        weight_path=args.dinov2_ckpt,
    ).to(device)
    model.enable_retrain_mode()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.decoder_parameters(), "lr": args.lr_decoder},
            {"params": model.encoder_bias_parameters(), "lr": args.lr_encoder_bias},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = SILogLoss()
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    step, total_loss, best_metrics, best_d1 = _load_candidate_checkpoint(
        args, arch_idx, model, optimizer, scaler, device
    )
    if is_main and step > 0:
        logger.info("Resumed candidate %d from step=%d", arch_idx, step)

    model.train()
    epoch = 0
    nan_hit = False
    consec_nan = 0
    while step < args.max_steps and not nan_hit:
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(args.seed + arch_idx * 100000 + epoch)
        for batch in train_loader:
            step += 1
            images = batch["image"].to(device, non_blocking=True)
            gt_depth = batch["depth"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                out = model(images)
                pred = out["depth"]
                if pred.shape[-2:] != gt_depth.shape[-2:]:
                    pred = F.interpolate(pred, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(pred, gt_depth)
            if not torch.isfinite(loss):
                # Transient non-finite loss (e.g. AMP fp16 overflow): skip this batch
                # and keep training. Only abort if it persists for many consecutive
                # steps (true divergence) -- a single late spike must NOT discard a
                # candidate that already reached a good best_d1.
                consec_nan += 1
                optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    "Candidate %d non-finite loss at step=%d (consecutive=%d/%d); skipping batch.",
                    arch_idx, step, consec_nan, args.nan_patience,
                )
                if consec_nan >= args.nan_patience:
                    logger.warning(
                        "Candidate %d diverged (%d consecutive non-finite); aborting arch.",
                        arch_idx, consec_nan,
                    )
                    nan_hit = True
                    break
                continue
            consec_nan = 0
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            total_loss += float(loss.item())
            should_validate = (
                args.val_every_steps > 0
                and (step % args.val_every_steps == 0 or step >= args.max_steps)
            )
            if should_validate:
                if is_ddp:
                    dist.barrier()
                metrics = evaluate(_unwrap_model(model), val_loader, device, args, is_main)
                if is_main:
                    metrics["train_loss"] = total_loss / max(step, 1)
                    metrics["best_step"] = step
                    if float(metrics["d1"]) > best_d1:
                        best_d1 = float(metrics["d1"])
                        best_metrics = dict(metrics)
                        logger.info(
                            "Candidate %d new best d1=%.4f at step=%d abs_rel=%.4f rmse=%.4f",
                            arch_idx,
                            best_d1,
                            step,
                            metrics["abs_rel"],
                            metrics["rmse"],
                        )
                    else:
                        logger.info(
                            "Candidate %d val step=%d d1=%.4f best_d1=%.4f abs_rel=%.4f rmse=%.4f",
                            arch_idx,
                            step,
                            metrics["d1"],
                            best_d1,
                            metrics["abs_rel"],
                            metrics["rmse"],
                        )
                if is_ddp:
                    dist.barrier()
                model.train()
            if (
                args.save_every_steps > 0
                and step % args.save_every_steps == 0
                and step < args.max_steps
            ):
                _save_candidate_checkpoint(
                    args,
                    arch_idx,
                    model,
                    optimizer,
                    scaler,
                    step,
                    total_loss,
                    best_metrics,
                    best_d1,
                    is_main,
                )
            if step >= args.max_steps:
                break
        epoch += 1

    if is_ddp:
        dist.barrier()
    if is_main and best_metrics:
        # A valid best exists. If a late NaN aborted training, keep it (early-stop)
        # rather than discarding a good architecture.
        if nan_hit:
            logger.info(
                "Candidate %d aborted by NaN but keeping best d1=%.4f (early-stopped).",
                arch_idx, best_d1,
            )
    elif is_main and nan_hit:
        # Truly degenerate: diverged before ever producing a valid evaluation.
        best_metrics = {"failed": True, "d1": float("nan"), "abs_rel": float("nan"),
                        "rmse": float("nan"), "train_loss": float("nan")}
        best_d1 = -float("inf")
    elif is_main and not best_metrics:
        metrics = evaluate(_unwrap_model(model), val_loader, device, args, is_main)
        metrics["train_loss"] = total_loss / max(step, 1)
        metrics["best_step"] = step
        best_metrics = dict(metrics)
        best_d1 = float(metrics["d1"])
    if is_ddp:
        dist.barrier()
    if is_main and best_metrics:
        best_metrics = dict(best_metrics)
        best_metrics["steps"] = step
        best_metrics.setdefault("best_step", step)
        best_metrics.setdefault("failed", False)
    if is_main:
        _save_candidate_checkpoint(
            args,
            arch_idx,
            model,
            optimizer,
            scaler,
            step,
            total_loss,
            best_metrics,
            best_d1,
            is_main,
        )
    return best_metrics


@torch.no_grad()
def evaluate(model: StudentNet, val_loader, device: torch.device, args, is_main: bool) -> Dict[str, float]:
    if not is_main:
        return {}
    model.eval()
    all_metrics: List[Dict[str, float]] = []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        gt_depth = batch["depth"].to(device, non_blocking=True)
        out = model(images)
        pred = out["depth"]
        if pred.shape[-2:] != gt_depth.shape[-2:]:
            pred = F.interpolate(pred, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(pred.shape[0]):
            all_metrics.append(
                compute_depth_metrics(
                    pred[i:i + 1],
                    gt_depth[i:i + 1],
                    min_depth=args.min_depth_eval,
                    max_depth=args.max_depth_eval,
                )
            )
    keys = list(all_metrics[0].keys())
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}


def load_completed_rows(out_dir: Path) -> List[Dict]:
    json_path = out_dir / "rank_consistency.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return list(data.get("rows", []))
    partial_path = out_dir / "rank_consistency_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            return list(json.load(f))
    return []


def write_outputs(out_dir: Path, rows: List[Dict], args):
    rows.sort(key=lambda r: r["global_order"])
    fieldnames = [
        "rank", "global_order", "idx", "proxy_score", "proxy_rank_grad", "grad", "grad_obj", "flops_g", "params_m",
        "d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "train_loss", "steps", "best_step", "arch",
    ]
    with open(out_dir / "rank_consistency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args) if rows else {}
    with open(out_dir / "rank_consistency.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    with open(out_dir / "rank_consistency_partial.json", "w") as f:
        json.dump(rows, f, indent=2)
    return summary


def _is_valid(r: Dict) -> bool:
    if r.get("failed"):
        return False
    try:
        d1 = float(r["d1"]); ar = float(r["abs_rel"])
    except (KeyError, ValueError, TypeError):
        return False
    return math.isfinite(d1) and d1 > 0 and math.isfinite(ar)


def summarize(rows: List[Dict], args) -> Dict:
    valid = [r for r in rows if _is_valid(r)]
    n_failed = len(rows) - len(valid)
    out = {"n_total": len(rows), "n_valid": len(valid), "n_failed": n_failed,
           "real_top_frac": args.real_top_frac}
    if len(valid) < 2:
        return out

    proxy = [float(r["proxy_score"]) for r in valid]
    # GSAP "small-grad-first" score that is higher-is-better, to correlate with
    # higher-is-better d1: proxy_score already encodes that ranking.
    d1 = [float(r["d1"]) for r in valid]
    neg_abs_rel = [-float(r["abs_rel"]) for r in valid]
    neg_rmse = [-float(r["rmse"]) for r in valid]
    k = min(args.hit_k, len(valid))
    # delta-1 saturates among good decoders; abs_rel/rmse are more discriminative,
    # so we report the proxy against all three real targets.
    out.update({
        "spearman_proxy_d1": spearman(proxy, d1),
        "kendall_proxy_d1": kendall_tau(proxy, d1),
        "spearman_proxy_neg_abs_rel": spearman(proxy, neg_abs_rel),
        "kendall_proxy_neg_abs_rel": kendall_tau(proxy, neg_abs_rel),
        "spearman_proxy_neg_rmse": spearman(proxy, neg_rmse),
        "hit_at_k_d1": hit_at_k(proxy, d1, k=k, real_top_frac=args.real_top_frac),
        "ndcg_at_k_d1": ndcg_at_k(proxy, d1, k=k),
        "hit_at_k_abs_rel": hit_at_k(proxy, neg_abs_rel, k=k, real_top_frac=args.real_top_frac),
        "ndcg_at_k_abs_rel": ndcg_at_k(proxy, neg_abs_rel, k=k),
        "hit_k": k,
    })
    return out


def main():
    args = parse_args()
    if args.dataset == "kitti":
        if args.depth_scale is None:
            args.depth_scale = 256.0
        if args.max_depth is None:
            args.max_depth = 80.0
        if args.max_depth_eval == 10.0:
            args.max_depth_eval = 80.0
        if args.img_h == 480 and args.img_w == 640:
            args.img_h, args.img_w = 352, 1216
    else:
        if args.depth_scale is None:
            args.depth_scale = 1000.0
        if args.max_depth is None:
            args.max_depth = 10.0
    local_rank, world_size, is_ddp = setup_ddp(args)
    is_main = local_rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s [rank{local_rank}] %(levelname)s %(message)s",
        force=True,
    )

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    candidates = load_candidates(args)
    train_loader, val_loader, train_sampler = build_loaders(args, is_ddp)
    if args.epochs is not None:
        steps_per_epoch = len(train_loader)
        args.max_steps = steps_per_epoch * args.epochs
        if is_main:
            logger.info(
                "epochs=%d -> max_steps=%d (steps/epoch=%d, per-rank batches)",
                args.epochs, args.max_steps, steps_per_epoch,
            )
    rows = load_completed_rows(out_dir) if is_main and args.resume else []
    completed_arches = {r["arch"] for r in rows}
    if is_ddp:
        obj = [completed_arches if is_main else None]
        dist.broadcast_object_list(obj, src=0)
        completed_arches = obj[0]
    if is_main:
        logger.info(
            "World size=%d candidates=%d completed=%d Train batches/rank=%d Val batches=%d",
            world_size,
            len(candidates),
            len(completed_arches),
            len(train_loader),
            len(val_loader),
        )

    for global_idx, candidate in enumerate(candidates):
        if candidate["arch"] in completed_arches:
            if is_main:
                logger.info("[%d/%d] skip completed arch=%s", global_idx + 1, len(candidates), candidate["arch"])
            continue
        if is_main:
            logger.info(
                "[%d/%d] proxy=%.4f grad=%.3e grad_obj=%.4f arch=%s",
                global_idx + 1,
                len(candidates),
                candidate["proxy_score"],
                candidate["grad"],
                candidate["grad_obj"],
                candidate["arch"],
            )
        metrics = train_and_eval_arch(
            args,
            candidate["arch"],
            global_idx,
            train_loader,
            train_sampler,
            val_loader,
            is_ddp,
            is_main,
        )
        if is_main:
            row = {**candidate, **metrics}
            row["global_order"] = global_idx
            row["rank"] = 0
            rows.append(row)
            completed_arches.add(candidate["arch"])
            logger.info(
                "best real d1=%.4f abs_rel=%.4f rmse=%.4f train_loss=%.4f best_step=%s",
                row["d1"],
                row["abs_rel"],
                row["rmse"],
                row["train_loss"],
                row.get("best_step", row.get("steps")),
            )
            summary = write_outputs(out_dir, rows, args)
            logger.info("Partial summary: %s", summary)

    if is_main:
        summary = write_outputs(out_dir, rows, args)
        logger.info("Summary: %s", summary)
        logger.info("Wrote %s", out_dir)
    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
