"""Metrics for locked anomaly-map evaluation without test-time tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def aupro(scores: np.ndarray, masks: np.ndarray, max_fpr: float = 0.30) -> float:
    """Compute AUPRO with Anomalib 2.6.0's official metric implementation."""
    import torch
    from anomalib.metrics.aupro import _AUPRO

    values = scores.astype(np.float32, copy=False)
    lower = float(np.min(values))
    upper = float(np.max(values))
    values = (values - lower) / max(upper - lower, 1e-12)
    metric = _AUPRO(fpr_limit=max_fpr)
    metric.update(
        torch.from_numpy(np.ascontiguousarray(values)),
        torch.from_numpy(np.ascontiguousarray((masks > 0).astype(np.int32))),
    )
    return float(metric.compute().item())


def evaluate_maps(
    scores: np.ndarray, masks: np.ndarray, threshold: float, *, include_aupro: bool = True
) -> dict[str, float]:
    labels = masks.reshape(-1).astype(np.uint8)
    flattened = scores.reshape(-1)
    image_labels = masks.reshape(len(masks), -1).max(axis=1).astype(np.uint8)
    image_scores = np.quantile(scores.reshape(len(scores), -1), 0.995, axis=1)
    metrics = {
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pixel_auroc": float(roc_auc_score(labels, flattened)),
        "pixel_f1": float(f1_score(labels, flattened >= threshold, zero_division=0)),
    }
    if include_aupro:
        metrics["aupro_30"] = aupro(scores, masks)
    return metrics
