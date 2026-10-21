"""Generate an MVTec-style synthetic calibration split from train/good only."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dafad.synthetic import generate_calibration_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--normal_dir")
    source.add_argument("--visa_root")
    parser.add_argument("--visa_split_csv", default="split_csv/1cls.csv")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--anomaly_count", type=int, default=32)
    parser.add_argument("--normal_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encode_normals_with_zero_masks", action="store_true")
    args = parser.parse_args()
    normal_images = None
    if args.visa_root:
        visa_root = Path(args.visa_root).resolve()
        split_csv = visa_root / args.visa_split_csv
        with split_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            normal_images = [
                visa_root / row["image"]
                for row in rows
                if row["object"] == args.category
                and row["split"].lower() == "train"
                and row["label"].lower() == "normal"
            ]
        if not normal_images:
            raise ValueError(f"No train/normal rows for {args.category} in {split_csv}")
        print(f"VisA train/normal calibration sources: {len(normal_images)}")

    output = generate_calibration_dataset(
        normal_dir=args.normal_dir,
        output_root=args.output_root,
        category=args.category,
        anomaly_count=args.anomaly_count,
        normal_count=args.normal_count,
        seed=args.seed,
        overwrite=args.overwrite,
        normal_images=normal_images,
        encode_normals_with_zero_masks=args.encode_normals_with_zero_masks,
    )
    print(f"synthetic calibration dataset: {output}")


if __name__ == "__main__":
    main()
