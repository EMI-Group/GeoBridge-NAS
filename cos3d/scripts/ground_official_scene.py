#!/usr/bin/env python3
"""Run COS3D kernel grounding for one official LERF-OVS scene.

This is a resumable version of Grounding_3D.py.  It replaces
PyTorch3D's Euclidean KNN call with scipy.cKDTree because PyTorch3D is not
available in the isolated runtime environment; both compute the same 10-nearest
spatial neighbours used by COS3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import open_clip
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


SCENE_QUERIES = {
    "figurines": (
        "jake",
        "pirate hat",
        "pikachu",
        "rubber duck with hat",
        "porcelain hand",
        "red apple",
        "tesla door handle",
        "waldo",
        "bag",
        "toy cat statue",
        "miffy",
        "green apple",
        "pumpkin",
        "rubics cube",
        "old camera",
        "rubber duck with buoy",
        "red toy chair",
        "pink ice cream",
        "spatula",
        "green toy chair",
        "toy elephant",
    ),
    "teatime": (
        "sheep",
        "yellow pouf",
        "stuffed bear",
        "coffee mug",
        "tea in a glass",
        "apple",
        "coffee",
        "hooves",
        "bear nose",
        "dall-e brand",
        "plate",
        "paper napkin",
        "three cookies",
        "bag of cookies",
    ),
    "ramen": (
        "nori",
        "sake cup",
        "kamaboko",
        "corn",
        "spoon",
        "egg",
        "onion segments",
        "plate",
        "napkin",
        "bowl",
        "glass of water",
        "hand",
        "chopsticks",
        "wavy noodles",
    ),
    "waldo_kitchen": (
        "Stainless steel pots",
        "dark cup",
        "refrigerator",
        "frog cup",
        "pot",
        "spatula",
        "plate",
        "spoon",
        "toaster",
        "ottolenghi",
        "plastic ladle",
        "sink",
        "ketchup",
        "cabinet",
        "red cup",
        "pour-over vessel",
        "knife",
        "yellow desk",
    ),
}
NEGATIVES = ("object", "things", "stuff", "texture")


def load_gaussian_fields(path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    ply = PlyData.read(path, mmap=True)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    semantic_names = [f"semantic_{index}" for index in range(16)]
    missing = [name for name in semantic_names if name not in names]
    if missing:
        raise ValueError(f"{path} has no complete 16D semantic field: {missing}")
    xyz = np.column_stack(
        [np.asarray(vertex[name]) for name in ("x", "y", "z")]
    ).astype(np.float32, copy=False)
    feature = np.column_stack(
        [np.asarray(vertex[name]) for name in semantic_names]
    ).astype(np.float32, copy=False)
    opacity = np.asarray(vertex["opacity"]).astype(np.float32, copy=False)
    return xyz, torch.from_numpy(feature.copy()), torch.from_numpy(opacity.copy())


@torch.no_grad()
def instance_to_language(
    feature: torch.Tensor,
    anchor_feature: torch.Tensor,
    anchor_clip: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    count = feature.shape[0]
    result = torch.empty(
        (count, anchor_clip.shape[1]),
        dtype=torch.float32,
        device=feature.device,
    )
    anchor_feature = F.normalize(anchor_feature.float(), dim=1, eps=1e-6)
    anchor_clip = F.normalize(anchor_clip.float(), dim=1, eps=1e-6)
    total_chunks = (count + chunk_size - 1) // chunk_size
    started = time.monotonic()
    for chunk_index, start in enumerate(range(0, count, chunk_size), start=1):
        stop = min(start + chunk_size, count)
        distances = 1.0 - feature[start:stop] @ anchor_feature.T
        nearest_distance, nearest_index = torch.topk(
            distances, 25, largest=False, dim=1
        )
        weights = torch.exp(-nearest_distance.square() / (2 * 0.1**2))
        # The official code applies one global scalar and normalizes each
        # output vector afterwards.  Row normalization is algebraically
        # equivalent and avoids underflow for large batches.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mapped = (anchor_clip[nearest_index] * weights[..., None]).sum(dim=1)
        result[start:stop] = F.normalize(mapped, dim=1, eps=1e-6)
        if chunk_index == 1 or chunk_index % 20 == 0 or stop == count:
            elapsed = time.monotonic() - started
            print(
                f"Instance2Language {stop}/{count} "
                f"chunks={chunk_index}/{total_chunks} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return result


@torch.no_grad()
def smooth_language_in_3d(
    xyz: np.ndarray,
    language: torch.Tensor,
    chunk_size: int,
    workers: int,
) -> torch.Tensor:
    print(f"Building cKDTree for {len(xyz)} Gaussians", flush=True)
    started = time.monotonic()
    tree = cKDTree(xyz, leafsize=32, compact_nodes=True, balanced_tree=True)
    print(f"cKDTree ready in {time.monotonic() - started:.1f}s", flush=True)
    total_chunks = (len(xyz) + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, len(xyz), chunk_size), start=1):
        stop = min(start + chunk_size, len(xyz))
        _, neighbour = tree.query(
            xyz[start:stop],
            k=10,
            workers=workers,
        )
        neighbour_tensor = torch.from_numpy(
            np.asarray(neighbour, dtype=np.int64)
        ).to(language.device)
        language[start:stop] = language[neighbour_tensor].mean(dim=1)
        language[start:stop] = F.normalize(
            language[start:stop], dim=1, eps=1e-6
        )
        if chunk_index == 1 or chunk_index % 20 == 0 or stop == len(xyz):
            print(
                f"Spatial smoothing {stop}/{len(xyz)} "
                f"chunks={chunk_index}/{total_chunks}",
                flush=True,
            )
    return language


def load_text_embeddings(
    checkpoint: Path,
    prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=str(checkpoint),
        precision="fp16",
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        positive_tokens = tokenizer(prompts).to(device)
        negative_tokens = tokenizer(list(NEGATIVES)).to(device)
        positive = F.normalize(
            model.encode_text(positive_tokens).float(), dim=1, eps=1e-6
        )
        negative = F.normalize(
            model.encode_text(negative_tokens).float(), dim=1, eps=1e-6
        )
    del model
    torch.cuda.empty_cache()
    return positive, negative


@torch.no_grad()
def relevancy_score(
    language: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    score = torch.empty(language.shape[0], dtype=torch.float32, device=language.device)
    for start in range(0, language.shape[0], chunk_size):
        stop = min(start + chunk_size, language.shape[0])
        current = language[start:stop].float()
        positive_value = current @ positive.T
        negative_value = current @ negative.T
        repeated_positive = positive_value.expand(-1, negative.shape[0])
        paired = torch.stack((repeated_positive, negative_value), dim=-1)
        probability = torch.softmax(10.0 * paired, dim=-1)[..., 0]
        score[start:stop] = probability.min(dim=1).values
    score = score - score.min()
    score = score / score.max().clamp_min(1e-9)
    return torch.clamp(score * 2.0 - 1.0, 0.0, 1.0)


@torch.no_grad()
def greedy_select(
    score: torch.Tensor,
    feature: torch.Tensor,
    opacity: torch.Tensor,
    mean_distance: float,
) -> tuple[torch.Tensor, int]:
    processed = torch.zeros_like(score, dtype=torch.bool)
    collected = torch.zeros_like(score, dtype=torch.bool)
    thresholds = [
        1.0 - factor * mean_distance for factor in (0.4, 0.3, 0.2, 0.1)
    ]
    groups = 0
    while True:
        candidates = (~processed) & (score > 0.5) & (opacity > 0.1)
        if not bool(candidates.any()):
            break
        candidate_score = torch.where(
            candidates, score, torch.full_like(score, -torch.inf)
        )
        seed = int(candidate_score.argmax())
        similarity = feature @ feature[seed]
        selected = None
        for threshold in thresholds:
            selected = (similarity > threshold) & (~processed)
            if not bool(selected.any()):
                processed[seed] = True
                selected = None
                break
            mean_score = (
                (score * opacity)[selected].sum()
                / opacity[selected].sum().clamp_min(1e-9)
            )
            if float(mean_score) >= 0.5:
                collected |= selected
                break
        if selected is not None:
            processed |= selected
        groups += 1
        if groups % 25 == 0:
            print(
                f"  greedy groups={groups} selected={int(collected.sum())}",
                flush=True,
            )
        if groups > 100_000:
            raise RuntimeError("Greedy grounding did not converge")
    return collected, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--scene", choices=tuple(SCENE_QUERIES), required=True
    )
    parser.add_argument("--clip-ckpt", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--spatial-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.spatial_workers < 1:
        raise ValueError("Chunk size and spatial workers must be positive")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    pair_path = args.exp_dir / "feature_CLIP_pair.pt"
    distance_path = args.exp_dir / "mean_distance.pt"
    ply_path = args.exp_dir / "point_cloud/iteration_30000/point_cloud.ply"
    required = [pair_path, distance_path, ply_path, args.clip_ckpt]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing grounding inputs:\n  " + "\n  ".join(missing))

    print(f"Loading Gaussian field from {ply_path}", flush=True)
    xyz, feature_cpu, opacity_cpu = load_gaussian_fields(ply_path)
    feature = F.normalize(feature_cpu.to(device), dim=1, eps=1e-6)
    opacity = torch.sigmoid(opacity_cpu.to(device))
    del feature_cpu, opacity_cpu
    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    anchor_feature = pair["feature"].to(device)
    anchor_clip = pair["clip"].to(device)
    mean_distance = float(
        torch.load(distance_path, map_location="cpu", weights_only=False)
    )
    print(
        f"Gaussians={feature.shape[0]} anchors={anchor_feature.shape[0]} "
        f"mean_distance={mean_distance:.6f}",
        flush=True,
    )

    language_path = args.exp_dir / "matching_CLIP_3D.pt"
    if language_path.exists() and not args.force:
        print(f"Loading cached {language_path}", flush=True)
        language = torch.load(
            language_path, map_location=device, weights_only=False
        ).float()
    else:
        language = instance_to_language(
            feature,
            anchor_feature,
            anchor_clip,
            args.chunk_size,
        )
        language = smooth_language_in_3d(
            xyz,
            language,
            args.chunk_size,
            args.spatial_workers,
        )
        print(f"Saving {language_path}", flush=True)
        torch.save(language.detach().cpu(), language_path)

    queries = SCENE_QUERIES[args.scene]
    prompts = list(queries)
    positive, negative = load_text_embeddings(args.clip_ckpt, prompts, device)
    mask_dir = args.exp_dir / "Grouding_3D_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "scene": args.scene,
        "gaussians": int(feature.shape[0]),
        "anchors": int(anchor_feature.shape[0]),
        "mean_distance": mean_distance,
        "queries": {},
    }
    for query_index, prompt in enumerate(queries):
        label = prompt
        mask_path = mask_dir / f"{label}.npy"
        if mask_path.exists() and not args.force:
            mask = np.load(mask_path)
            summary["queries"][label] = {
                "prompt": prompt,
                "selected": int(mask.sum()),
                "cached": True,
            }
            continue
        print(f"Grounding query: {prompt} -> {label}", flush=True)
        score = relevancy_score(
            language,
            positive[query_index : query_index + 1],
            negative,
            max(args.chunk_size, 65_536),
        )
        selected, groups = greedy_select(
            score, feature, opacity, mean_distance
        )
        selected_count = int(selected.sum())
        if selected_count == 0:
            print(
                f"WARNING: grounding produced an empty mask for {prompt}",
                flush=True,
            )
        np.save(mask_path, selected.cpu().numpy())
        summary["queries"][label] = {
            "prompt": prompt,
            "selected": selected_count,
            "groups": groups,
            "score_max": float(score.max()),
            "score_mean": float(score.mean()),
        }
        print(
            f"Saved {mask_path} selected={selected_count} groups={groups}",
            flush=True,
        )
    summary_path = args.exp_dir / "grounding_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Grounding complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
