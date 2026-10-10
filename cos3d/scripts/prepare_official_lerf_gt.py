#!/usr/bin/env python3
"""Build the official LERF-OVS ground-truth masks from polygon JSON files.

This intentionally mirrors LangSplat's evaluation preprocessing:
``np.int32`` polygon coordinates, ``cv2.fillPoly``, union by category, and
JPEG masks whose foreground value is 255.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


EXPECTED_COUNTS = {
    "figurines": 56,
    "teatime": 59,
    "ramen": 71,
    "waldo_kitchen": 22,
}


def polygon_to_mask(
    image_shape: tuple[int, int], points_list: list[list[float]]
) -> np.ndarray:
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base: np.ndarray, mask_add: np.ndarray) -> np.ndarray:
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def vis_mask_save(mask: np.ndarray, save_path: Path) -> None:
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    if not cv2.imwrite(str(save_path), mask_save):
        raise RuntimeError(f"cv2.imwrite failed: {save_path}")


def build_scene(scene_dir: Path) -> tuple[int, int]:
    json_files = sorted(scene_dir.glob("frame_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No frame_*.json annotations in {scene_dir}")

    gt_dir = scene_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as handle:
            gt_data = json.load(handle)

        info = gt_data["info"]
        height = int(info["height"])
        width = int(info["width"])
        frame_name = Path(info["name"]).stem
        if frame_name != json_path.stem:
            raise ValueError(
                f"Frame mismatch: {json_path.name} says {info['name']!r}"
            )

        annotated_image = cv2.imread(str(scene_dir / info["name"]))
        if annotated_image is None:
            raise FileNotFoundError(scene_dir / info["name"])
        if annotated_image.shape[:2] != (height, width):
            raise ValueError(
                f"Shape mismatch for {info['name']}: "
                f"JSON={(height, width)}, image={annotated_image.shape[:2]}"
            )

        masks: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((height, width), dtype=np.uint8)
        )
        for obj in gt_data["objects"]:
            category = obj["category"]
            if Path(category).name != category:
                raise ValueError(f"Unsafe category name {category!r}")
            polygon = polygon_to_mask((height, width), obj["segmentation"])
            masks[category] = stack_mask(masks[category], polygon)

        frame_dir = gt_dir / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        for category, mask in sorted(masks.items()):
            if not np.any(mask):
                raise ValueError(
                    f"Empty polygon mask: {scene_dir.name}/{frame_name}/{category}"
                )
            vis_mask_save(mask, frame_dir / f"{category}.jpg")
            written += 1

    return len(json_files), written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-root",
        type=Path,
        required=True,
        help="Directory containing figurines/teatime/ramen/waldo_kitchen labels",
    )
    args = parser.parse_args()

    total = 0
    for scene, expected in EXPECTED_COUNTS.items():
        frames, written = build_scene(args.label_root / scene)
        if written != expected:
            raise RuntimeError(
                f"{scene}: generated {written} masks, expected {expected}"
            )
        total += written
        print(f"{scene}: frames={frames}, masks={written}")

    if total != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError(f"Generated {total} masks, expected 208")
    print(f"OK: generated {total} official LERF-OVS masks")


if __name__ == "__main__":
    main()
