#!/usr/bin/env python3
"""Fuse aligned DepthSplat pair predictions into one scene-level 3DGS model.

Unlike the old merger, this script never ranks Gaussians by opacity.  It
spatially de-duplicates pair predictions in world coordinates, averages the
attributes of predictions that fall in the same voxel, resets opacity to the
standard 3DGS initialization value, and matches the Gaussian capacity of the
corresponding COS3D baseline scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from merge_depthsplat_gaussians import DEFAULT_SCENES, SH_REST_DIM, _load_vertices


BASELINE_CAPACITY = {
    "figurines": 927_155,
    "teatime": 1_868_657,
    "ramen": 569_408,
    "waldo_kitchen": 1_669_684,
}
FUSED_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def _first_pass(
    paths: list[Path],
    sample_per_file: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect a bounded sample used only for robust bounds and voxel scale."""
    rng = np.random.default_rng(seed)
    xyz_samples: list[np.ndarray] = []
    scale_samples: list[np.ndarray] = []
    input_points = 0
    for path in paths:
        fields = _load_vertices(path)
        count = fields["x"].size
        input_points += count
        finite = np.ones(count, dtype=bool)
        for name in FUSED_FIELDS:
            finite &= np.isfinite(fields[name])
        candidates = np.flatnonzero(finite)
        if candidates.size > sample_per_file:
            candidates = rng.choice(
                candidates, size=sample_per_file, replace=False
            )
        xyz_samples.append(
            np.stack(
                (
                    fields["x"][candidates],
                    fields["y"][candidates],
                    fields["z"][candidates],
                ),
                axis=1,
            ).astype(np.float32, copy=False)
        )
        scale_samples.append(
            np.exp(
                np.stack(
                    (
                        fields["scale_0"][candidates],
                        fields["scale_1"][candidates],
                        fields["scale_2"][candidates],
                    ),
                    axis=1,
                )
            ).astype(np.float32, copy=False)
        )
    return (
        np.concatenate(xyz_samples, axis=0),
        np.concatenate(scale_samples, axis=0),
        input_points,
    )


def _robust_bounds(
    xyz_sample: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.quantile(xyz_sample, quantile, axis=0)
    upper = np.quantile(xyz_sample, 1.0 - quantile, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    # Keep a margin around the robust interval so that legitimate scene edges
    # are not removed along with the extreme pair-prediction outliers.
    return lower - 0.05 * span, upper + 0.05 * span


def _load_filtered_fields(
    paths: list[Path],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    chunks = {name: [] for name in FUSED_FIELDS}
    finite_points = 0
    bounded_points = 0
    for path in paths:
        fields = _load_vertices(path)
        count = fields["x"].size
        keep = np.ones(count, dtype=bool)
        for name in FUSED_FIELDS:
            keep &= np.isfinite(fields[name])
        finite_points += int(keep.sum())
        keep &= fields["x"] >= lower[0]
        keep &= fields["x"] <= upper[0]
        keep &= fields["y"] >= lower[1]
        keep &= fields["y"] <= upper[1]
        keep &= fields["z"] >= lower[2]
        keep &= fields["z"] <= upper[2]
        bounded_points += int(keep.sum())
        for name in FUSED_FIELDS:
            chunks[name].append(
                np.asarray(fields[name][keep], dtype=np.float32)
            )
    return (
        {name: np.concatenate(values) for name, values in chunks.items()},
        finite_points,
    )


def _voxel_inverse(
    xyz: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, int]:
    quantized = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    quantized -= quantized.min(axis=0)
    dimensions = quantized.max(axis=0) + 1
    if (
        int(dimensions[0])
        * int(dimensions[1])
        * int(dimensions[2])
        >= np.iinfo(np.int64).max
    ):
        raise RuntimeError(
            "Voxel grid is too large to encode safely; increase voxel size"
        )
    keys = quantized[:, 0]
    keys += dimensions[0] * (
        quantized[:, 1] + dimensions[1] * quantized[:, 2]
    )
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int32, copy=False), int(inverse.max()) + 1


def _choose_voxelization(
    xyz: np.ndarray,
    origin: np.ndarray,
    initial_voxel_size: float,
    target: int,
    maximum_rounds: int,
) -> tuple[np.ndarray, int, float, list[dict]]:
    voxel_size = max(float(initial_voxel_size), 1e-6)
    history = []
    best: tuple[np.ndarray, int, float] | None = None
    for _ in range(maximum_rounds):
        inverse, groups = _voxel_inverse(xyz, origin, voxel_size)
        history.append({"voxel_size": voxel_size, "occupied_voxels": groups})
        if groups >= target:
            best = (inverse, groups, voxel_size)
        if target <= groups <= int(target * 1.75):
            return inverse, groups, voxel_size, history
        if groups < target:
            voxel_size *= 0.72
            continue
        # Occupied voxels on scene surfaces scale closer to 1/v^2 than 1/v^3.
        ratio = groups / (target * 1.35)
        voxel_size *= float(np.clip(np.sqrt(ratio), 1.12, 2.0))
    if best is not None:
        return best[0], best[1], best[2], history
    # The scale estimate can only be too coarse here. Refine until enough
    # occupied cells exist, which guarantees the requested baseline capacity.
    for _ in range(8):
        voxel_size *= 0.65
        inverse, groups = _voxel_inverse(xyz, origin, voxel_size)
        history.append({"voxel_size": voxel_size, "occupied_voxels": groups})
        if groups >= target:
            return inverse, groups, voxel_size, history
    raise RuntimeError(
        f"Only {groups} occupied voxels for requested capacity {target}"
    )


def _select_and_remap_groups(
    inverse: np.ndarray,
    group_count: int,
    target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    if group_count == target:
        selected = np.arange(group_count, dtype=np.int32)
    else:
        selected = np.sort(
            rng.choice(group_count, size=target, replace=False)
        ).astype(np.int32)
    selected_flags = np.zeros(group_count, dtype=bool)
    selected_flags[selected] = True
    point_mask = selected_flags[inverse]
    remap = np.full(group_count, -1, dtype=np.int32)
    remap[selected] = np.arange(target, dtype=np.int32)
    selected_inverse = remap[inverse[point_mask]]
    return point_mask, selected_inverse, int(point_mask.sum())


def _mean_by_group(
    value: np.ndarray,
    point_mask: np.ndarray,
    selected_inverse: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    sums = np.bincount(
        selected_inverse,
        weights=np.asarray(value[point_mask], dtype=np.float64),
        minlength=counts.size,
    )
    return (sums / counts).astype(np.float32)


def _write_ply(
    output_path: Path,
    xyz: np.ndarray,
    f_dc: np.ndarray,
    opacity_logit: np.ndarray,
    scale_log: np.ndarray,
    rotation: np.ndarray,
) -> None:
    count = xyz.shape[0]
    normals = np.zeros_like(xyz, dtype=np.float32)
    f_rest = np.zeros((count, SH_REST_DIM), dtype=np.float32)
    attributes = np.concatenate(
        (
            xyz,
            normals,
            f_dc,
            f_rest,
            opacity_logit[:, None],
            scale_log,
            rotation,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(SH_REST_DIM)]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    elements = np.empty(count, dtype=[(name, "f4") for name in names])
    for index, name in enumerate(names):
        elements[name] = attributes[:, index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)


def _fuse_scene(
    source_root: Path,
    output_root: Path,
    scene: str,
    expected_pairs: int,
    target: int,
    initial_opacity: float,
    trim_quantile: float,
    sample_per_file: int,
    voxel_scale_factor: float,
    maximum_voxel_rounds: int,
    seed: int,
) -> dict:
    paths = sorted(source_root.glob(f"worker_*/gaussians/{scene}__pair_*.ply"))
    if len(paths) != expected_pairs:
        raise RuntimeError(
            f"{scene}: found {len(paths)} pair PLYs, expected {expected_pairs}"
        )
    xyz_sample, scale_sample, input_points = _first_pass(
        paths, sample_per_file, seed
    )
    lower, upper = _robust_bounds(xyz_sample, trim_quantile)
    fields, finite_points = _load_filtered_fields(paths, lower, upper)
    xyz = np.stack((fields["x"], fields["y"], fields["z"]), axis=1)
    median_scale = float(np.median(np.max(scale_sample, axis=1)))
    inverse, group_count, voxel_size, history = _choose_voxelization(
        xyz,
        lower,
        median_scale * voxel_scale_factor,
        target,
        maximum_voxel_rounds,
    )
    point_mask, selected_inverse, selected_point_count = (
        _select_and_remap_groups(inverse, group_count, target, seed)
    )
    counts = np.bincount(selected_inverse, minlength=target).astype(np.float64)

    fused_xyz = np.stack(
        [
            _mean_by_group(fields[name], point_mask, selected_inverse, counts)
            for name in ("x", "y", "z")
        ],
        axis=1,
    )
    fused_f_dc = np.stack(
        [
            _mean_by_group(fields[name], point_mask, selected_inverse, counts)
            for name in ("f_dc_0", "f_dc_1", "f_dc_2")
        ],
        axis=1,
    )
    fused_scale_log = np.stack(
        [
            _mean_by_group(fields[name], point_mask, selected_inverse, counts)
            for name in ("scale_0", "scale_1", "scale_2")
        ],
        axis=1,
    )
    scale = np.exp(fused_scale_log)
    scale = np.clip(scale, voxel_size * 0.05, voxel_size * 2.0)
    fused_scale_log = np.log(scale).astype(np.float32)

    rotation_source = np.stack(
        [fields[f"rot_{index}"] for index in range(4)], axis=1
    )
    rotation_source[rotation_source[:, 0] < 0.0] *= -1.0
    fused_rotation = np.stack(
        [
            _mean_by_group(
                rotation_source[:, index],
                point_mask,
                selected_inverse,
                counts,
            )
            for index in range(4)
        ],
        axis=1,
    )
    norm = np.linalg.norm(fused_rotation, axis=1, keepdims=True)
    fused_rotation /= np.maximum(norm, 1e-8)
    invalid_rotation = norm[:, 0] < 1e-8
    fused_rotation[invalid_rotation] = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    fused_opacity = np.full(target, _logit(initial_opacity), dtype=np.float32)
    output_path = (
        output_root / scene / "point_cloud" / "iteration_0" / "point_cloud.ply"
    )
    _write_ply(
        output_path,
        fused_xyz,
        fused_f_dc,
        fused_opacity,
        fused_scale_log,
        fused_rotation,
    )
    return {
        "scene": scene,
        "pair_files": len(paths),
        "input_points": input_points,
        "finite_points": finite_points,
        "bounded_points": int(xyz.shape[0]),
        "trim_quantile": trim_quantile,
        "sample_median_max_scale": median_scale,
        "voxel_scale_factor": voxel_scale_factor,
        "voxel_size": voxel_size,
        "occupied_voxels": group_count,
        "selected_source_points": selected_point_count,
        "target_baseline_capacity": target,
        "output_points": int(fused_xyz.shape[0]),
        "initial_opacity": initial_opacity,
        "voxel_search": history,
        "xyz_min": fused_xyz.min(axis=0).tolist(),
        "xyz_max": fused_xyz.max(axis=0).tolist(),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-opacity", type=float, default=0.1)
    parser.add_argument("--trim-quantile", type=float, default=0.001)
    parser.add_argument("--sample-per-file", type=int, default=4096)
    parser.add_argument("--voxel-scale-factor", type=float, default=2.0)
    parser.add_argument("--maximum-voxel-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    expected = {
        item["scene"]: int(item["generated_pairs"])
        for item in manifest["scenes"]
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene in DEFAULT_SCENES:
        print(f"[fusion] {scene}: starting", flush=True)
        summary = _fuse_scene(
            args.source_root,
            args.output_root,
            scene,
            expected[scene],
            BASELINE_CAPACITY[scene],
            args.initial_opacity,
            args.trim_quantile,
            args.sample_per_file,
            args.voxel_scale_factor,
            args.maximum_voxel_rounds,
            args.seed,
        )
        summaries.append(summary)
        (args.output_root / "generation_summary.partial.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[fusion] {scene}: {input_points} raw -> {bounded_points} bounded "
            "-> {occupied_voxels} voxels -> {output_points} Gaussians".format(
                **summary
            ),
            flush=True,
        )
    (args.output_root / "generation_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print("[fusion] all scenes complete", flush=True)


if __name__ == "__main__":
    main()
