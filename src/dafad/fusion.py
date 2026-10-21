"""Test-independent calibration and fusion for two anomaly-map branches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class FusionCalibration:
    efficient_lower: float
    efficient_upper: float
    patchcore_lower: float
    patchcore_upper: float
    efficient_weight: float
    efficient_threshold: float
    patchcore_threshold: float
    threshold: float
    validation_pixel_auroc: float
    validation_pixel_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> "FusionCalibration":
        return cls(**value)

    def fuse(self, efficient_maps: np.ndarray, patchcore_maps: np.ndarray) -> np.ndarray:
        efficient = normalize(efficient_maps, self.efficient_lower, self.efficient_upper)
        patchcore = normalize(patchcore_maps, self.patchcore_lower, self.patchcore_upper)
        return self.efficient_weight * efficient + (1.0 - self.efficient_weight) * patchcore


def normalize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    scale = max(upper - lower, 1e-12)
    return np.clip((values.astype(np.float64) - lower) / scale, 0.0, 1.0)


def _best_threshold(scores: np.ndarray, labels: np.ndarray, grid_size: int) -> tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    positive = np.sort(scores[labels == 1])
    negative = np.sort(scores[labels == 0])
    true_positive = len(positive) - np.searchsorted(positive, thresholds, side="left")
    false_positive = len(negative) - np.searchsorted(negative, thresholds, side="left")
    false_negative = len(positive) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    index = int(np.argmax(values))
    return float(thresholds[index]), float(values[index])


def _sample_pixels(
    efficient: np.ndarray,
    patchcore: np.ndarray,
    labels: np.ndarray,
    max_normal_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(negative) > max_normal_pixels:
        rng = np.random.default_rng(seed)
        negative = rng.choice(negative, size=max_normal_pixels, replace=False)
    selected = np.concatenate((positive, negative))
    return efficient[selected], patchcore[selected], labels[selected]


def fit_fusion_profile(
    efficient_maps: np.ndarray,
    patchcore_maps: np.ndarray,
    masks: np.ndarray,
    candidate_weights: list[float],
    lower_quantile: float = 0.50,
    upper_quantile: float = 0.995,
    threshold_grid_size: int = 201,
    max_normal_pixels: int = 1_000_000,
    sampling_seed: int = 0,
) -> dict[str, object]:
    """Fit category normalization and calibration curves for every candidate weight."""
    if efficient_maps.shape != patchcore_maps.shape or efficient_maps.shape != masks.shape:
        raise ValueError("EfficientAD maps, PatchCore maps, and masks must have identical shapes")
    labels = (masks.reshape(-1) > 0).astype(np.uint8)
    if np.unique(labels).size != 2:
        raise ValueError("Calibration masks must contain both normal and anomalous pixels")
    e_lower, e_upper = np.quantile(efficient_maps, [lower_quantile, upper_quantile])
    p_lower, p_upper = np.quantile(patchcore_maps, [lower_quantile, upper_quantile])
    efficient = normalize(efficient_maps, float(e_lower), float(e_upper)).reshape(-1)
    patchcore = normalize(patchcore_maps, float(p_lower), float(p_upper)).reshape(-1)
    efficient_sample, patchcore_sample, sampled_labels = _sample_pixels(
        efficient,
        patchcore,
        labels,
        max_normal_pixels=max_normal_pixels,
        seed=sampling_seed,
    )
    curves: dict[str, dict[str, float]] = {}
    for weight in sorted({float(value) for value in candidate_weights}):
        scores = weight * efficient_sample + (1.0 - weight) * patchcore_sample
        threshold, f1 = _best_threshold(scores, sampled_labels, threshold_grid_size)
        curves[f"{weight:.1f}"] = {
            "pixel_auroc": float(roc_auc_score(sampled_labels, scores)),
            "best_pixel_f1": f1,
            "threshold": threshold,
        }
    efficient_threshold, _ = _best_threshold(efficient_sample, sampled_labels, threshold_grid_size)
    patchcore_threshold, _ = _best_threshold(patchcore_sample, sampled_labels, threshold_grid_size)
    return {
        "efficient_lower": float(e_lower),
        "efficient_upper": float(e_upper),
        "patchcore_lower": float(p_lower),
        "patchcore_upper": float(p_upper),
        "efficient_threshold": efficient_threshold,
        "patchcore_threshold": patchcore_threshold,
        "candidate_curves": curves,
    }


def calibration_from_profile(profile: dict[str, object]) -> FusionCalibration:
    """Build the category-adaptive calibration from a previously fitted profile."""
    curves = profile["candidate_curves"]
    selected_weight = min(
        (float(weight) for weight in curves),
        key=lambda weight: (
            -curves[f"{weight:.1f}"]["pixel_auroc"],
            -curves[f"{weight:.1f}"]["best_pixel_f1"],
            weight,
        ),
    )
    selected = curves[f"{selected_weight:.1f}"]
    return FusionCalibration(
        efficient_lower=profile["efficient_lower"],
        efficient_upper=profile["efficient_upper"],
        patchcore_lower=profile["patchcore_lower"],
        patchcore_upper=profile["patchcore_upper"],
        efficient_weight=selected_weight,
        efficient_threshold=profile["efficient_threshold"],
        patchcore_threshold=profile["patchcore_threshold"],
        threshold=selected["threshold"],
        validation_pixel_auroc=selected["pixel_auroc"],
        validation_pixel_f1=selected["best_pixel_f1"],
    )


def fit_fusion(
    efficient_maps: np.ndarray,
    patchcore_maps: np.ndarray,
    masks: np.ndarray,
    candidate_weights: list[float],
    lower_quantile: float = 0.50,
    upper_quantile: float = 0.995,
    threshold_grid_size: int = 201,
    max_normal_pixels: int = 1_000_000,
    sampling_seed: int = 0,
) -> FusionCalibration:
    profile = fit_fusion_profile(
        efficient_maps,
        patchcore_maps,
        masks,
        candidate_weights,
        lower_quantile,
        upper_quantile,
        threshold_grid_size,
        max_normal_pixels,
        sampling_seed,
    )
    return calibration_from_profile(profile)
