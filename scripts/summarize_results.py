"""Aggregate per-category evaluation metrics with an unweighted macro mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    categories = {}
    for filename in args.results:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        category = payload.get("category") or Path(filename).name.split("_fusion_")[0]
        if category in categories:
            raise ValueError(f"Duplicate category result: {category}")
        categories[category] = payload
    methods = set.intersection(*(set(item["methods"]) for item in categories.values()))
    summary = {
        method: {
            metric: sum(item["methods"][method][metric] for item in categories.values()) / len(categories)
            for metric in ("image_auroc", "pixel_auroc", "pixel_f1", "aupro_30")
        }
        for method in sorted(methods)
    }
    output_payload = {
        "category_count": len(categories),
        "aggregation": "unweighted macro-average across categories",
        "summary": summary,
        "categories": {
            category: {"methods": item["methods"], "fusion_weights": item["fusion_weights"]}
            for category, item in categories.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps({"category_count": len(categories), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
