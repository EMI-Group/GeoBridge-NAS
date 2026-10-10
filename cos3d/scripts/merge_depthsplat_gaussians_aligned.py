#!/usr/bin/env python3
"""Merge eight aligned A3 worker exports with equal budget per view pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from merge_depthsplat_gaussians import (
    DEFAULT_SCENES,
    SH_REST_DIM,
    _load_vertices,
    _sigmoid,
)


def _merge_scene(
    source_root: Path,
    output_root: Path,
    scene: str,
    expected_pairs: int,
    opacity_threshold: float,
    max_points: int,
) -> dict:
    paths = sorted(
        source_root.glob(f"worker_*/gaussians/{scene}__pair_*.ply")
    )
    if len(paths) != expected_pairs:
        raise RuntimeError(
            f"{scene}: found {len(paths)} aligned pair PLYs, "
            f"expected {expected_pairs}"
        )
    base_budget, remainder = divmod(max_points, len(paths))
    selected_fields: dict[str, list[np.ndarray]] = {}
    input_points = 0
    finite_points = 0
    retained_per_pair = []

    for pair_index, path in enumerate(paths):
        fields = _load_vertices(path)
        input_points += fields["x"].size
        finite = np.ones(fields["x"].shape[0], dtype=bool)
        for value in fields.values():
            finite &= np.isfinite(value)
        finite_points += int(finite.sum())
        opacity_probability = _sigmoid(fields["opacity"])
        candidates = np.flatnonzero(
            finite & (opacity_probability >= opacity_threshold)
        )
        budget = base_budget + (1 if pair_index < remainder else 0)
        if candidates.size > budget:
            scores = opacity_probability[candidates]
            local = np.argpartition(scores, -budget)[-budget:]
            indices = candidates[local]
        else:
            indices = candidates
        retained_per_pair.append(int(indices.size))
        for name, values in fields.items():
            selected_fields.setdefault(name, []).append(values[indices])

    fields = {
        name: np.concatenate(chunks) for name, chunks in selected_fields.items()
    }
    count = fields["x"].size
    xyz = np.stack((fields["x"], fields["y"], fields["z"]), axis=1)
    normals = np.zeros_like(xyz)
    f_dc = np.stack(
        (fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]), axis=1
    )
    f_rest = np.zeros((count, SH_REST_DIM), dtype=np.float32)
    opacity = fields["opacity"][:, None]
    scale = np.stack(
        (fields["scale_0"], fields["scale_1"], fields["scale_2"]), axis=1
    )
    rotation = np.stack(
        tuple(fields[f"rot_{index}"] for index in range(4)), axis=1
    )
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacity, scale, rotation), axis=1
    ).astype(np.float32, copy=False)

    property_names = ["x", "y", "z", "nx", "ny", "nz"]
    property_names += [f"f_dc_{index}" for index in range(3)]
    property_names += [f"f_rest_{index}" for index in range(SH_REST_DIM)]
    property_names += ["opacity"]
    property_names += [f"scale_{index}" for index in range(3)]
    property_names += [f"rot_{index}" for index in range(4)]
    elements = np.empty(count, dtype=[(name, "f4") for name in property_names])
    elements[:] = list(map(tuple, attributes))

    output_path = (
        output_root / scene / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    return {
        "scene": scene,
        "pair_files": len(paths),
        "input_points": input_points,
        "finite_points": finite_points,
        "opacity_threshold": opacity_threshold,
        "max_points": max_points,
        "output_points": int(count),
        "minimum_points_per_pair": min(retained_per_pair),
        "maximum_points_per_pair": max(retained_per_pair),
        "output": str(output_path.resolve()),
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=1_500_000)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    expected = {
        item["scene"]: int(item["generated_pairs"])
        for item in manifest["scenes"]
    }
    summaries = [
        _merge_scene(
            args.source_root,
            args.output_root,
            scene,
            expected[scene],
            args.opacity_threshold,
            args.max_points,
        )
        for scene in DEFAULT_SCENES
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            "{scene}: {input_points} -> {output_points} points from "
            "{pair_files} aligned A3 predictions".format(**summary)
        )


if __name__ == "__main__":
    main()
