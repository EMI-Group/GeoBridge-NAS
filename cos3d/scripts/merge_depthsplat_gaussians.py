#!/usr/bin/env python3
"""Merge world-coordinate DepthSplat PLYs into COS3D-compatible 3DGS PLYs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


DEFAULT_SCENES = ("figurines", "teatime", "ramen", "waldo_kitchen")
SH_REST_DIM = 45  # 3 * ((degree 3 + 1)^2 - 1)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _load_vertices(path: Path) -> dict[str, np.ndarray]:
    vertices = PlyData.read(path)["vertex"]
    names = vertices.data.dtype.names
    required = (
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
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is missing PLY properties: {missing}")
    return {
        name: np.asarray(vertices[name], dtype=np.float32) for name in required
    }


def _merge_scene(
    source_root: Path,
    output_root: Path,
    scene: str,
    opacity_threshold: float,
    max_points: int,
) -> dict:
    paths = sorted((source_root / scene / "gaussians").glob(f"{scene}__pair_*.ply"))
    if not paths:
        raise FileNotFoundError(
            f"No pair PLY files found under {source_root / scene / 'gaussians'}"
        )

    loaded = [_load_vertices(path) for path in paths]
    fields = {
        name: np.concatenate([item[name] for item in loaded])
        for name in loaded[0]
    }
    finite = np.ones(fields["x"].shape[0], dtype=bool)
    for value in fields.values():
        finite &= np.isfinite(value)
    opacity_probability = _sigmoid(fields["opacity"])
    keep = finite & (opacity_probability >= opacity_threshold)
    indices = np.flatnonzero(keep)
    if max_points > 0 and indices.size > max_points:
        scores = opacity_probability[indices]
        top = np.argpartition(scores, -max_points)[-max_points:]
        indices = indices[top]
        indices = indices[np.argsort(scores[top])[::-1]]

    xyz = np.stack(
        (fields["x"][indices], fields["y"][indices], fields["z"][indices]), axis=1
    )
    normals = np.zeros_like(xyz)
    f_dc = np.stack(
        (
            fields["f_dc_0"][indices],
            fields["f_dc_1"][indices],
            fields["f_dc_2"][indices],
        ),
        axis=1,
    )
    f_rest = np.zeros((indices.size, SH_REST_DIM), dtype=np.float32)
    opacity = fields["opacity"][indices, None]
    scale = np.stack(
        (
            fields["scale_0"][indices],
            fields["scale_1"][indices],
            fields["scale_2"][indices],
        ),
        axis=1,
    )
    rotation = np.stack(
        tuple(fields[f"rot_{index}"][indices] for index in range(4)), axis=1
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
    dtype = [(name, "f4") for name in property_names]
    elements = np.empty(indices.size, dtype=dtype)
    elements[:] = list(map(tuple, attributes))

    output_path = (
        output_root / scene / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    return {
        "scene": scene,
        "pair_files": len(paths),
        "input_points": int(finite.size),
        "finite_points": int(finite.sum()),
        "opacity_threshold": opacity_threshold,
        "output_points": int(indices.size),
        "output": str(output_path.resolve()),
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "mean_opacity": float(opacity_probability[indices].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=1_500_000)
    args = parser.parse_args()

    summaries = [
        _merge_scene(
            args.source_root,
            args.output_root,
            scene,
            args.opacity_threshold,
            args.max_points,
        )
        for scene in args.scenes
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            "{scene}: {input_points} -> {output_points} points from "
            "{pair_files} A3 predictions".format(**summary)
        )


if __name__ == "__main__":
    main()
