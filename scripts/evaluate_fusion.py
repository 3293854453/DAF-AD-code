"""Evaluate the frozen calibration comparison on aligned test maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dafad.ensemble import RobustStats, robust_standardize  # noqa: E402
from dafad.fusion import normalize  # noqa: E402
from dafad.metrics import evaluate_maps  # noqa: E402

EXPECTED_UNLOCK_TOKEN = "FROZEN_DAFAD_V1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--calibration", required=True, help="Category JSON or dataset-global JSON")
    parser.add_argument("--category", help="Required when --calibration is a dataset-global JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlock_token", required=True)
    args = parser.parse_args()
    if args.unlock_token != EXPECTED_UNLOCK_TOKEN:
        raise PermissionError("Invalid test unlock token; calibrate and freeze parameters before evaluation")

    values = np.load(args.maps)
    saved = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if "categories" in saved:
        if not args.category:
            raise ValueError("--category is required when using dataset-global calibration")
        if args.category not in saved["categories"]:
            raise KeyError(f"Category {args.category!r} is absent from dataset calibration")
        category = saved["categories"][args.category]
        global_weight = float(saved["selected_efficient_weight"])
    else:
        category = saved
        global_weight = None

    efficient_raw = values["efficient"]
    patchcore_raw = values["patchcore"]
    masks = values["masks"]
    adaptive = category["adaptive"]
    efficient = normalize(efficient_raw, adaptive["efficient_lower"], adaptive["efficient_upper"])
    patchcore = normalize(patchcore_raw, adaptive["patchcore_lower"], adaptive["patchcore_upper"])
    if global_weight is None:
        # Legacy category-only calibration remains readable.
        global_weight = float(adaptive["efficient_weight"])
        fixed_weight = 0.5
    else:
        fixed_weight = 0.5

    global_cal = category["global"] if "global" in category else None
    adaptive_cal = adaptive
    fixed_cal = category["fixed"] if "fixed" in category else None
    branch_cal = category.get("branches", {})
    mad_cal = category.get("mad_mean")
    if fixed_cal is None or global_cal is None or mad_cal is None:
        raise ValueError(
            "Calibration JSON must include fixed/global profiles and MAD-mean calibration; "
            "regenerate it with the current scripts"
        )

    fused_fixed = fixed_weight * efficient + (1.0 - fixed_weight) * patchcore
    fused_global = global_weight * efficient + (1.0 - global_weight) * patchcore
    fused_adaptive = adaptive_cal["efficient_weight"] * efficient + (
        1.0 - adaptive_cal["efficient_weight"]
    ) * patchcore
    mad = (
        robust_standardize(efficient_raw, RobustStats(**mad_cal["efficient_stats"]))
        + robust_standardize(patchcore_raw, RobustStats(**mad_cal["patchcore_stats"]))
    ) * 0.5

    methods = {
        "efficient_ad": evaluate_maps(efficient, masks, branch_cal["efficient_threshold"]),
        "patchcore": evaluate_maps(patchcore, masks, branch_cal["patchcore_threshold"]),
        "mad_mean": evaluate_maps(mad, masks, mad_cal["threshold"]),
        "fixed": evaluate_maps(fused_fixed, masks, fixed_cal["threshold"]),
        "dataset_global": evaluate_maps(fused_global, masks, global_cal["threshold"]),
        "category_adaptive": evaluate_maps(fused_adaptive, masks, adaptive_cal["threshold"]),
    }
    payload = {
        "category": args.category,
        "methods": methods,
        "fusion_weights": {
            "fixed_efficient": fixed_weight,
            "dataset_global_efficient": global_weight,
            "category_adaptive_efficient": float(adaptive_cal["efficient_weight"]),
        },
        "thresholds": {
            "efficient_ad": float(branch_cal["efficient_threshold"]),
            "patchcore": float(branch_cal["patchcore_threshold"]),
            "mad_mean": float(mad_cal["threshold"]),
            "fixed": float(fixed_cal["threshold"]),
            "dataset_global": float(global_cal["threshold"]),
            "category_adaptive": float(adaptive_cal["threshold"]),
        },
        "aupro_protocol": "Anomalib 2.6.0 _AUPRO, fpr_limit=0.3; scores min-max scaled monotonically per category/method",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
