"""Select one dataset-global fusion weight from category calibration profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrations", nargs="+", required=True, metavar="CATEGORY=JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    categories: dict[str, dict] = {}
    for entry in args.calibrations:
        if "=" not in entry:
            raise ValueError(f"Expected CATEGORY=JSON, got: {entry}")
        category, filename = entry.split("=", 1)
        if category in categories:
            raise ValueError(f"Duplicate calibration category: {category}")
        categories[category] = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not categories:
        raise ValueError("At least one category calibration is required")

    candidate_sets = [set(item["profile"]["candidate_curves"]) for item in categories.values()]
    if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
        raise ValueError("All categories must use the same candidate weights")
    weights = sorted(float(value) for value in candidate_sets[0])
    macro_curves = {}
    for weight in weights:
        key = f"{weight:.1f}"
        macro_curves[key] = {
            "macro_pixel_auroc": sum(
                item["profile"]["candidate_curves"][key]["pixel_auroc"]
                for item in categories.values()
            ) / len(categories),
            "macro_best_pixel_f1": sum(
                item["profile"]["candidate_curves"][key]["best_pixel_f1"]
                for item in categories.values()
            ) / len(categories),
        }
    selected_weight = min(
        weights,
        key=lambda weight: (
            -macro_curves[f"{weight:.1f}"]["macro_pixel_auroc"],
            -macro_curves[f"{weight:.1f}"]["macro_best_pixel_f1"],
            weight,
        ),
    )
    key = f"{selected_weight:.1f}"
    payload = {
        "selection": "macro-average category calibration pixel AUROC; macro-average best pixel F1 breaks ties; lowest weight breaks remaining ties",
        "selected_efficient_weight": selected_weight,
        "selected_patchcore_weight": 1.0 - selected_weight,
        "macro_candidate_curves": macro_curves,
        "categories": {
            category: {
                "adaptive": item["adaptive"],
                "profile": item["profile"],
                "fixed": item["fixed"],
                "global": item["profile"]["candidate_curves"][key],
                "branches": item["branches"],
                "mad_mean": item["mad_mean"],
            }
            for category, item in categories.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected_efficient_weight": selected_weight, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
