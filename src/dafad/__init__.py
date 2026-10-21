"""DAF-AD adaptive anomaly-map fusion."""

from .fusion import FusionCalibration, fit_fusion
from .metrics import aupro, evaluate_maps
from .synthetic import generate_calibration_dataset

__all__ = [
    "FusionCalibration",
    "aupro",
    "evaluate_maps",
    "fit_fusion",
    "generate_calibration_dataset",
]
