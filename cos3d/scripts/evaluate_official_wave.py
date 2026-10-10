#!/usr/bin/env python3
"""Evaluate one single-seed, four-scene method wave."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


SCENE_FRAMES = {
    "figurines": (
        "frame_00041",
        "frame_00105",
        "frame_00152",
        "frame_00195",
    ),
    "teatime": (
        "frame_00002",
        "frame_00025",
        "frame_00043",
        "frame_00107",
        "frame_00129",
        "frame_00140",
    ),
    "ramen": (
        "frame_00006",
        "frame_00024",
        "frame_00060",
        "frame_00065",
        "frame_00081",
        "frame_00119",
        "frame_00128",
    ),
    "waldo_kitchen": (
        "frame_00053",
        "frame_00066",
        "frame_00089",
        "frame_00140",
        "frame_00154",
    ),
}
EXPECTED_PAIRS = {
    "figurines": 56,
    "teatime": 59,
    "ramen": 71,
    "waldo_kitchen": 22,
}


def binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L")) > 10).astype(np.uint8)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shape mismatch: {mask_a.shape} versus {mask_b.shape}"
        )
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def evaluate_scene(
    gt_root: Path, pred_root: Path, scene: str
) -> dict[str, object]:
    values: list[float] = []
    missing: list[str] = []
    pairs: list[dict[str, object]] = []
    for frame in SCENE_FRAMES[scene]:
        frame_root = gt_root / frame
        gt_files = sorted(
            path
            for path in frame_root.iterdir()
            if path.suffix.lower() in {".jpg", ".png"}
        )
        for gt_path in gt_files:
            pred_path = pred_root / f"{frame}_{gt_path.stem}.png"
            if pred_path.is_file():
                value = iou(binary_mask(gt_path), binary_mask(pred_path))
            else:
                value = 0.0
                missing.append(pred_path.name)
            values.append(value)
            pairs.append(
                {
                    "frame": frame,
                    "query": gt_path.stem,
                    "iou": value,
                    "prediction": str(pred_path),
                    "missing": not pred_path.is_file(),
                }
            )
    if len(values) != EXPECTED_PAIRS[scene]:
        raise RuntimeError(
            f"{scene}: evaluated {len(values)} pairs, "
            f"expected {EXPECTED_PAIRS[scene]}"
        )
    array = np.asarray(values)
    return {
        "pairs": len(values),
        "missing_predictions": missing,
        "miou": float(array.mean()),
        "acc_025": float((array > 0.25).mean()),
        "acc_050": float((array > 0.50).mean()),
        "details": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    args = parser.parse_args()

    launch_manifest = args.method_root / "launch_manifest.json"
    launch_data = json.loads(launch_manifest.read_text(encoding="utf-8"))
    runs = launch_data["runs"]
    if len(runs) != 4:
        raise RuntimeError(f"Expected 4 formal runs, found {len(runs)}")

    output: dict[str, object] = {
        "protocol": {
            "metric": "released COS3D LERF mIoU / Acc@0.25 / Acc@0.5",
            "threshold": 10,
            "paper_mAcc_mapping": "Acc@0.25",
            "scene_aggregation": "unweighted macro average",
        },
        "runs": {},
        "seeds": {},
    }
    csv_rows: list[dict[str, object]] = []
    for run in runs:
        scene = str(run["scene"])
        seed = int(run["seed"])
        run_name = Path(run["output"]).name
        run_output = Path(run["output"])
        if not run_output.is_absolute():
            run_output = args.method_root / run_output
        pred_root = (
            run_output
            / "text2obj_prompt_Grouding_3D_mask"
            / "ours_30000"
            / "renders_cluster_silhouette"
        )
        gt_root = args.label_root / scene / "gt"
        if not pred_root.is_dir() or not gt_root.is_dir():
            raise FileNotFoundError(
                f"Missing evaluation directory for {run_name}: "
                f"gt={gt_root}, pred={pred_root}"
            )
        metrics = evaluate_scene(gt_root, pred_root, scene)
        output["runs"][run_name] = {
            "scene": scene,
            "seed": seed,
            **metrics,
        }
        csv_rows.append(
            {
                "run": run_name,
                "scene": scene,
                "seed": seed,
                "mIoU": 100.0 * metrics["miou"],
                "Acc@0.25": 100.0 * metrics["acc_025"],
                "Acc@0.5": 100.0 * metrics["acc_050"],
                "missing": len(metrics["missing_predictions"]),
            }
        )

    for seed in sorted({int(run["seed"]) for run in runs}):
        seed_metrics = [
            metrics
            for metrics in output["runs"].values()
            if metrics["seed"] == seed
        ]
        if len(seed_metrics) != 4:
            raise RuntimeError(f"Seed {seed}: expected four scenes")
        summary = {
            key: float(np.mean([metrics[key] for metrics in seed_metrics]))
            for key in ("miou", "acc_025", "acc_050")
        }
        output["seeds"][str(seed)] = summary
        csv_rows.append(
            {
                "run": f"macro_seed{seed}",
                "scene": "Average",
                "seed": seed,
                "mIoU": 100.0 * summary["miou"],
                "Acc@0.25": 100.0 * summary["acc_025"],
                "Acc@0.5": 100.0 * summary["acc_050"],
                "missing": sum(
                    len(metrics["missing_predictions"])
                    for metrics in seed_metrics
                ),
            }
        )

    output["mean"] = {
        key: float(
            np.mean(
                [seed_metrics[key] for seed_metrics in output["seeds"].values()]
            )
        )
        for key in ("miou", "acc_025", "acc_050")
    }
    json_path = args.method_root / "official_metrics.json"
    csv_path = args.method_root / "official_metrics.csv"
    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output["mean"], indent=2))
    print(f"json={json_path.resolve()}")
    print(f"csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
