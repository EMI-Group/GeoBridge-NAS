#!/usr/bin/env python3
"""Build eight balanced DepthSplat workers covering every COS3D train view.

The train/test split is exactly the split used by COS3D's COLMAP reader:
``index % 8 != 2`` for training.  Sliding pairs retain DepthSplat's two-view
interface while ensuring that every baseline training image is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from prepare_depthsplat_lerf import (
    DEFAULT_SCENES,
    _camera_row,
    _encoded_image,
    _read_colmap,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    parser.add_argument("--pair-gap", type=int, default=6)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.workers != 8:
        raise ValueError("The aligned experiment requires exactly eight workers")
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_examples: list[list[dict]] = [[] for _ in range(args.workers)]
    worker_indices: list[dict[str, str]] = [{} for _ in range(args.workers)]
    worker_evaluations: list[dict[str, dict[str, list[int]]]] = [
        {} for _ in range(args.workers)
    ]
    worker_pairs: list[list[dict]] = [[] for _ in range(args.workers)]
    scene_manifests = []
    global_pair_index = 0

    for scene in args.scenes:
        scene_dir = args.source_root / scene
        extrinsics, intrinsics = _read_colmap(scene_dir)
        ordered = sorted(extrinsics.values(), key=lambda item: Path(item.name).stem)
        train_views = [
            image
            for index, image in enumerate(ordered)
            if index % args.llffhold != 2
        ]
        gap = min(max(1, args.pair_gap), len(train_views) - 1)
        pair_starts = range(len(train_views) - gap)
        coverage: Counter[str] = Counter()
        scene_pairs = []

        for pair_index, left in enumerate(pair_starts):
            right = left + gap
            views = (train_views[left], train_views[right])
            key = f"{scene}__pair_{pair_index:04d}"
            worker = global_pair_index % args.workers
            global_pair_index += 1
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
                coverage[image_path.name] += 1

            worker_examples[worker].append(
                {
                    "key": key,
                    "cameras": torch.stack(camera_rows),
                    "images": encoded_images,
                }
            )
            worker_indices[worker][key] = "000000.torch"
            worker_evaluations[worker][key] = {"context": [0, 1], "target": [0]}
            pair_record = {
                "key": key,
                "scene": scene,
                "worker": worker,
                "train_indices": [left, right],
                "images": image_names,
            }
            worker_pairs[worker].append(pair_record)
            scene_pairs.append(pair_record)

        train_names = [
            (scene_dir / "images" / Path(image.name).name).name
            for image in train_views
        ]
        missing = [name for name in train_names if coverage[name] == 0]
        if missing:
            raise RuntimeError(f"{scene}: uncovered baseline train views: {missing}")
        scene_manifests.append(
            {
                "scene": scene,
                "source": str(scene_dir.resolve()),
                "total_views": len(ordered),
                "baseline_train_views": len(train_views),
                "baseline_test_views": len(ordered) - len(train_views),
                "split_rule": f"sorted_index % {args.llffhold} != 2",
                "pair_gap": gap,
                "generated_pairs": len(scene_pairs),
                "covered_train_views": len(coverage),
                "minimum_view_coverage": min(coverage.values()),
                "maximum_view_coverage": max(coverage.values()),
                "train_images": train_names,
                "pairs": scene_pairs,
            }
        )

    for worker in range(args.workers):
        worker_root = args.output_root / f"worker_{worker}"
        test_root = worker_root / "test"
        test_root.mkdir(parents=True, exist_ok=True)
        torch.save(worker_examples[worker], test_root / "000000.torch")
        (test_root / "index.json").write_text(
            json.dumps(worker_indices[worker], indent=2) + "\n",
            encoding="utf-8",
        )
        (worker_root / "evaluation_index.json").write_text(
            json.dumps(worker_evaluations[worker], indent=2) + "\n",
            encoding="utf-8",
        )
        (worker_root / "manifest.json").write_text(
            json.dumps(
                {
                    "worker": worker,
                    "examples": len(worker_examples[worker]),
                    "pairs": worker_pairs[worker],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "protocol": "COS3D-aligned-all-train-views",
        "workers": args.workers,
        "total_pairs": global_pair_index,
        "worker_examples": [len(items) for items in worker_examples],
        "scenes": scene_manifests,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    for scene in scene_manifests:
        print(
            "{scene}: {covered_train_views}/{baseline_train_views} train views, "
            "{generated_pairs} pairs, coverage={minimum_view_coverage}.."
            "{maximum_view_coverage}".format(**scene)
        )
    print(
        f"workers={args.workers} total_pairs={global_pair_index} "
        f"distribution={manifest['worker_examples']}"
    )


if __name__ == "__main__":
    main()
