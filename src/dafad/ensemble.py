"""Training-side robust calibration for post-hoc anomaly-map ensembles."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RobustStats:
    median: float
    scaled_mad: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normal_reference_indices(
    masks: np.ndarray,
    max_pixels: int = 1_000_000,
    seed: int = 0,
) -> tuple[np.ndarray, str, int]:
    """Select calibration-side normal pixels without consulting test labels."""
    if masks.ndim < 2 or len(masks) == 0:
        raise ValueError("Calibration masks must contain at least one image")
    flattened_masks = masks.reshape(len(masks), -1)
    normal_images = flattened_masks.max(axis=1) == 0
    if not normal_images.any():
        raise ValueError("MAD-mean requires clean calibration images with zero masks")
    pixels_per_image = flattened_masks.shape[1]
    image_indices = np.flatnonzero(normal_images)
    indices = (
        image_indices[:, None] * pixels_per_image
        + np.arange(pixels_per_image, dtype=np.int64)[None, :]
    ).reshape(-1)
    source = "clean_calibration_images"
    total = int(len(indices))
    if total == 0:
        raise ValueError("Calibration maps contain no normal reference pixels")
    if total > max_pixels:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_pixels, replace=False))
    return indices.astype(np.int64, copy=False), source, total


def fit_robust_stats(values: np.ndarray, indices: np.ndarray) -> RobustStats:
    reference = values.reshape(-1)[indices].astype(np.float64)
    median = float(np.median(reference))
    scaled_mad = float(1.4826 * np.median(np.abs(reference - median)))
    return RobustStats(median=median, scaled_mad=max(scaled_mad, 1e-8))


def robust_standardize(values: np.ndarray, stats: RobustStats) -> np.ndarray:
    return (values.astype(np.float64) - stats.median) / stats.scaled_mad


def aggregate_maps(
    efficient: np.ndarray,
    patchcore: np.ndarray,
    operator: str,
) -> np.ndarray:
    if efficient.shape != patchcore.shape:
        raise ValueError("EfficientAD and PatchCore maps must have identical shapes")
    stacked = np.stack((efficient, patchcore), axis=0)
    if operator == "mean":
        return stacked.mean(axis=0)
    if operator == "median":
        return np.median(stacked, axis=0)
    if operator == "max":
        return stacked.max(axis=0)
    raise ValueError(f"Unsupported ensemble operator: {operator}")


def best_f1_threshold(
    scores: np.ndarray,
    masks: np.ndarray,
    grid_size: int = 201,
    max_normal_pixels: int = 1_000_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Fit a frozen pixel threshold on calibration labels only."""
    if scores.shape != masks.shape:
        raise ValueError("Scores and masks must have identical shapes")
    labels = (masks.reshape(-1) > 0).astype(np.uint8)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Calibration masks must contain normal and anomalous pixels")
    if len(negative) > max_normal_pixels:
        rng = np.random.default_rng(seed)
        negative = rng.choice(negative, size=max_normal_pixels, replace=False)
    values = scores.reshape(-1).astype(np.float64)
    positive_values = np.sort(values[positive])
    negative_values = np.sort(values[negative])
    sampled = np.concatenate((positive_values, negative_values))
    thresholds = np.unique(np.quantile(sampled, np.linspace(0.0, 1.0, grid_size)))
    true_positive = len(positive_values) - np.searchsorted(positive_values, thresholds, side="left")
    false_positive = len(negative_values) - np.searchsorted(negative_values, thresholds, side="left")
    false_negative = len(positive_values) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])

