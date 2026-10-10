#!/usr/bin/env python3
"""Convert posed LERF/COLMAP scenes into small DepthSplat test chunks.

Each generated example contains one context pair in the original COLMAP world
coordinate system.  DepthSplat predicts a complete Gaussian set for every
pair; the companion merge script combines those sets into one scene-level
3DGS backbone for COS3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


DEFAULT_SCENES = ("figurines", "teatime", "ramen", "waldo_kitchen")


def _read_colmap(scene_dir: Path):
    sparse = scene_dir / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
    return extrinsics, intrinsics


def _normalized_intrinsics(camera) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float32)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = float(params[0])
        cx, cy = map(float, params[1:3])
    elif camera.model in {"PINHOLE", "OPENCV"}:
        fx, fy, cx, cy = map(float, params[:4])
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {camera.model!r}; "
            "use the undistorted PINHOLE images."
        )
    return (
        fx / float(camera.width),
        fy / float(camera.height),
        cx / float(camera.width),
        cy / float(camera.height),
    )


def _camera_row(image, camera) -> torch.Tensor:
    fx, fy, cx, cy = _normalized_intrinsics(camera)
    world_to_camera = np.concatenate(
        (
            qvec2rotmat(image.qvec).astype(np.float32),
            np.asarray(image.tvec, dtype=np.float32)[:, None],
        ),
        axis=1,
    )
    row = np.zeros(18, dtype=np.float32)
    row[:4] = (fx, fy, cx, cy)
    row[6:] = world_to_camera.reshape(-1)
    return torch.from_numpy(row)


def _encoded_image(path: Path) -> torch.Tensor:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
    return torch.from_numpy(data)


def _pair_starts(num_views: int, num_pairs: int, gap: int) -> list[int]:
    if num_views < 2:
        raise ValueError("At least two training views are required")
    gap = min(max(1, gap), num_views - 1)
    last = num_views - gap - 1
    starts = np.linspace(0, last, num=min(num_pairs, last + 1))
    starts = np.unique(np.rint(starts).astype(np.int64))
    return starts.tolist()


def _write_scene(
    scene_dir: Path,
    output_root: Path,
    num_pairs: int,
    pair_gap: int,
    llffhold: int,
) -> dict:
    extrinsics, intrinsics = _read_colmap(scene_dir)
    ordered = sorted(extrinsics.values(), key=lambda item: Path(item.name).stem)
    train_views = [
        image for index, image in enumerate(ordered) if index % llffhold != 2
    ]
    starts = _pair_starts(len(train_views), num_pairs, pair_gap)
    gap = min(max(1, pair_gap), len(train_views) - 1)

    examples = []
    evaluation_index = {}
    pair_manifest = []
    for pair_index, left in enumerate(starts):
        right = left + gap
        views = (train_views[left], train_views[right])
        key = f"{scene_dir.name}__pair_{pair_index:02d}"
        camera_rows = []
        encoded_images = []
        image_names = []
        for image in views:
            camera = intrinsics[image.camera_id]
            image_path = scene_dir / "images" / Path(image.name).name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            camera_rows.append(_camera_row(image, camera))
            encoded_images.append(_encoded_image(image_path))
            image_names.append(image_path.name)

        examples.append(
            {
                "key": key,
                "cameras": torch.stack(camera_rows),
                "images": encoded_images,
            }
        )
        evaluation_index[key] = {"context": [0, 1], "target": [0]}
        pair_manifest.append(
            {
                "key": key,
                "train_indices": [left, right],
                "images": image_names,
            }
        )

    scene_output = output_root / scene_dir.name
    test_dir = scene_output / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = "000000.torch"
    torch.save(examples, test_dir / chunk_name)
    (test_dir / "index.json").write_text(
        json.dumps({item["key"]: chunk_name for item in examples}, indent=2),
        encoding="utf-8",
    )
    (scene_output / "evaluation_index.json").write_text(
        json.dumps(evaluation_index, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "scene": scene_dir.name,
        "source": str(scene_dir.resolve()),
        "train_views": len(train_views),
        "llffhold": llffhold,
        "requested_pairs": num_pairs,
        "generated_pairs": len(examples),
        "pair_gap": gap,
        "pairs": pair_manifest,
    }
    (scene_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_scenes(values: Iterable[str]) -> tuple[str, ...]:
    scenes = tuple(values)
    return scenes if scenes else DEFAULT_SCENES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--pair-gap", type=int, default=6)
    parser.add_argument("--llffhold", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for scene in _parse_scenes(args.scenes):
        manifests.append(
            _write_scene(
                args.source_root / scene,
                args.output_root,
                args.pairs,
                args.pair_gap,
                args.llffhold,
            )
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifests, indent=2),
        encoding="utf-8",
    )
    for manifest in manifests:
        print(
            "{scene}: {generated_pairs} pairs from {train_views} train views".format(
                **manifest
            )
        )


if __name__ == "__main__":
    main()
