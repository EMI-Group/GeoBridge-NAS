"""Shared preregistered depth metrics for qualitative-sample selection."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Dict, Tuple

import numpy as np


EPSILON = 1e-6
MIN_DEPTH = 1e-3
EDGE_LOG_THRESHOLD = float(np.log(1.05))


def maximum_depth(dataset: str) -> float:
    if dataset == "nyu":
        return 10.0
    if dataset == "kitti":
        return 80.0
    raise ValueError(dataset)


def evaluation_mask(dataset: str, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    if dataset == "nyu":
        if shape != (480, 640):
            raise ValueError(f"Expected NYU shape (480, 640), got {shape}")
        mask[45:471, 41:601] = True
    elif dataset == "kitti":
        if shape != (352, 1216):
            raise ValueError(f"Expected KITTI shape (352, 1216), got {shape}")
        mask[
            int(0.40810811 * height) : int(0.99189189 * height),
            int(0.03594771 * width) : int(0.96405229 * width),
        ] = True
    else:
        raise ValueError(dataset)
    return mask


def sanitize_prediction(prediction: np.ndarray, dataset: str) -> np.ndarray:
    maximum = maximum_depth(dataset)
    output = np.asarray(prediction, dtype=np.float32).copy()
    output[np.isposinf(output)] = maximum
    output[np.isneginf(output)] = MIN_DEPTH
    output[np.isnan(output)] = MIN_DEPTH
    return np.clip(output, MIN_DEPTH, maximum)


def valid_depth_mask(ground_truth: np.ndarray, dataset: str) -> np.ndarray:
    maximum = maximum_depth(dataset)
    ground_truth = np.asarray(ground_truth)
    return (
        np.isfinite(ground_truth)
        & (ground_truth > MIN_DEPTH)
        & (ground_truth < maximum)
        & evaluation_mask(dataset, ground_truth.shape)
    )


def edge_pair_payload(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    dataset: str,
) -> Dict[str, np.ndarray]:
    prediction = sanitize_prediction(prediction, dataset)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    valid = valid_depth_mask(ground_truth, dataset)
    log_prediction = np.log(np.maximum(prediction, EPSILON))
    log_ground_truth = np.log(np.maximum(ground_truth, EPSILON))

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal_gt_gradient = log_ground_truth[:, 1:] - log_ground_truth[:, :-1]
    vertical_gt_gradient = log_ground_truth[1:, :] - log_ground_truth[:-1, :]
    horizontal_edge = horizontal_valid & (
        np.abs(horizontal_gt_gradient) >= EDGE_LOG_THRESHOLD
    )
    vertical_edge = vertical_valid & (
        np.abs(vertical_gt_gradient) >= EDGE_LOG_THRESHOLD
    )
    horizontal_pred_gradient = log_prediction[:, 1:] - log_prediction[:, :-1]
    vertical_pred_gradient = log_prediction[1:, :] - log_prediction[:-1, :]
    horizontal_error = np.abs(horizontal_pred_gradient - horizontal_gt_gradient)
    vertical_error = np.abs(vertical_pred_gradient - vertical_gt_gradient)
    return {
        "valid": valid,
        "horizontal_edge": horizontal_edge,
        "vertical_edge": vertical_edge,
        "horizontal_error": horizontal_error,
        "vertical_error": vertical_error,
    }


def per_sample_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    dataset: str,
) -> Dict[str, float | int]:
    prediction = sanitize_prediction(prediction, dataset)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    payload = edge_pair_payload(prediction, ground_truth, dataset)
    valid = payload["valid"]
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError("Sample has no valid evaluation pixels")
    horizontal_edge = payload["horizontal_edge"]
    vertical_edge = payload["vertical_edge"]
    edge_pairs = int(horizontal_edge.sum() + vertical_edge.sum())
    if edge_pairs == 0:
        gradient_mae = float("nan")
    else:
        gradient_mae = float(
            (
                payload["horizontal_error"][horizontal_edge].sum(dtype=np.float64)
                + payload["vertical_error"][vertical_edge].sum(dtype=np.float64)
            )
            / edge_pairs
        )
    abs_rel = float(
        np.mean(
            np.abs(prediction[valid] - ground_truth[valid]) / ground_truth[valid],
            dtype=np.float64,
        )
    )
    return {
        "valid_pixels": valid_pixels,
        "edge_pairs": edge_pairs,
        "abs_rel": abs_rel,
        "log_depth_edge_gmae": gradient_mae,
    }


def scene_or_sequence(sample_id: str, dataset: str) -> str:
    parts = PurePosixPath(sample_id).parts
    if dataset == "nyu":
        if len(parts) < 2:
            raise ValueError(sample_id)
        return parts[0]
    if dataset == "kitti":
        if len(parts) < 2:
            raise ValueError(sample_id)
        return parts[1]
    raise ValueError(dataset)
