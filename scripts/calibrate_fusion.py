"""Fit category fusion settings and MAD-mean statistics from calibration maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dafad.ensemble import fit_robust_stats, normal_reference_indices, robust_standardize, best_f1_threshold  # noqa: E402
from dafad.fusion import calibration_from_profile, fit_fusion_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", required=True, help="NPZ with efficient, patchcore, and masks arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate_weights",
        nargs="+",
        type=float,
        default=[i / 10 for i in range(11)],
        help="EfficientAD weights to compare; PatchCore receives the remaining weight",
    )
    parser.add_argument("--lower_quantile", type=float, default=0.50)
    parser.add_argument("--upper_quantile", type=float, default=0.995)
    parser.add_argument("--threshold_grid_size", type=int, default=201)
    parser.add_argument("--max_normal_pixels", type=int, default=1_000_000)
    parser.add_argument("--sampling_seed", type=int, default=0)
    args = parser.parse_args()
    values = np.load(args.maps)
    efficient_maps = values["efficient"]
    patchcore_maps = values["patchcore"]
    masks = values["masks"]
    profile = fit_fusion_profile(
        efficient_maps,
        patchcore_maps,
        masks,
        candidate_weights=args.candidate_weights,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        threshold_grid_size=args.threshold_grid_size,
        max_normal_pixels=args.max_normal_pixels,
        sampling_seed=args.sampling_seed,
    )
    calibration = calibration_from_profile(profile)
    normal_indices, reference_source, reference_count = normal_reference_indices(
        masks, max_pixels=args.max_normal_pixels, seed=args.sampling_seed
    )
    efficient_stats = fit_robust_stats(efficient_maps, normal_indices)
    patchcore_stats = fit_robust_stats(patchcore_maps, normal_indices)
    mad_mean = (
        robust_standardize(efficient_maps, efficient_stats)
        + robust_standardize(patchcore_maps, patchcore_stats)
    ) * 0.5
    mad_threshold, mad_f1 = best_f1_threshold(
        mad_mean, masks, grid_size=args.threshold_grid_size,
        max_normal_pixels=args.max_normal_pixels, seed=args.sampling_seed,
    )
    payload = {
        "adaptive": calibration.to_dict(),
        "profile": profile,
        "fixed": profile["candidate_curves"]["0.5"],
        "branches": {
            "efficient_threshold": profile["efficient_threshold"],
            "patchcore_threshold": profile["patchcore_threshold"],
        },
        "mad_mean": {
            "efficient_stats": efficient_stats.to_dict(),
            "patchcore_stats": patchcore_stats.to_dict(),
            "threshold": mad_threshold,
            "calibration_pixel_f1": mad_f1,
            "normal_reference_source": reference_source,
            "normal_reference_pixels_before_cap": reference_count,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
