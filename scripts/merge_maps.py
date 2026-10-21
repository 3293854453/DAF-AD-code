"""Merge two model-specific NPZ files after strict path and mask checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficient", required=True)
    parser.add_argument("--patchcore", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    efficient = np.load(args.efficient)
    patchcore = np.load(args.patchcore)
    if not np.array_equal(efficient["paths"], patchcore["paths"]):
        raise ValueError("EfficientAD and PatchCore image paths are not aligned")
    if not np.array_equal(efficient["masks"], patchcore["masks"]):
        raise ValueError("EfficientAD and PatchCore masks are not identical")
    if efficient["maps"].shape != patchcore["maps"].shape:
        raise ValueError("EfficientAD and PatchCore map shapes differ")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        efficient=efficient["maps"],
        patchcore=patchcore["maps"],
        masks=efficient["masks"],
        paths=efficient["paths"],
    )
    print(f"merged maps: {output.resolve()}")


if __name__ == "__main__":
    main()
