"""Deterministic synthetic calibration data built only from normal images."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _polygon_mask(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    cx = rng.randint(width // 6, max(width // 6, width * 5 // 6))
    cy = rng.randint(height // 6, max(height // 6, height * 5 // 6))
    rx = rng.randint(max(4, width // 25), max(5, width // 7))
    ry = rng.randint(max(4, height // 25), max(5, height // 7))
    points: list[tuple[int, int]] = []
    count = rng.randint(6, 11)
    for index in range(count):
        angle = 6.283185307179586 * index / count
        radial = rng.uniform(0.65, 1.15)
        x = int(cx + rx * radial * __import__("math").cos(angle))
        y = int(cy + ry * radial * __import__("math").sin(angle))
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def _scratch_mask(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(1, 3)):
        x0 = rng.randint(0, max(0, width - 1))
        y0 = rng.randint(0, max(0, height - 1))
        length = rng.randint(max(8, width // 12), max(9, width // 3))
        x1 = max(0, min(width - 1, x0 + rng.randint(-length, length)))
        y1 = max(0, min(height - 1, y0 + rng.randint(-length, length)))
        draw.line((x0, y0, x1, y1), fill=255, width=rng.randint(2, max(3, width // 80)))
    return mask


def synthesize(target: Image.Image, donor: Image.Image, rng: random.Random, mode: str) -> tuple[Image.Image, Image.Image]:
    target = target.convert("RGB")
    donor = ImageOps.fit(donor.convert("RGB"), target.size)
    mask = _scratch_mask(target.size, rng) if mode == "scratch" else _polygon_mask(target.size, rng)

    if mode == "dark":
        altered = ImageEnhance.Brightness(target).enhance(rng.uniform(0.15, 0.55))
    elif mode == "bright":
        altered = ImageEnhance.Contrast(ImageEnhance.Brightness(target).enhance(rng.uniform(1.3, 1.9))).enhance(1.4)
    elif mode == "scratch":
        altered = ImageOps.colorize(ImageOps.grayscale(target), black="#101010", white="#f2f2f2")
    else:
        altered = ImageEnhance.Color(ImageEnhance.Contrast(donor).enhance(rng.uniform(1.4, 2.2))).enhance(0.35)

    alpha = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.0)))
    synthetic = Image.composite(altered, target, alpha)
    binary_mask = mask.point(lambda value: 255 if value > 0 else 0)
    return synthetic, binary_mask


def generate_calibration_dataset(
    normal_dir: str | Path | None,
    output_root: str | Path,
    category: str,
    anomaly_count: int = 32,
    normal_count: int = 16,
    seed: int = 0,
    overwrite: bool = False,
    normal_images: Iterable[str | Path] | None = None,
    encode_normals_with_zero_masks: bool = False,
) -> Path:
    output_root = Path(output_root).resolve()
    category_root = output_root / category
    resolved_normal_dir = Path(normal_dir).resolve() if normal_dir is not None else None
    if normal_images is None:
        if resolved_normal_dir is None:
            raise ValueError("normal_dir or normal_images must be provided")
        source_images = _images(resolved_normal_dir)
    else:
        source_images = sorted(
            Path(path).resolve()
            for path in normal_images
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        )
    if len(source_images) < 2:
        raise ValueError("At least two normal images are required for calibration")
    missing = [path for path in source_images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calibration source image: {missing[0]}")
    if category_root.exists():
        if not overwrite:
            raise FileExistsError(f"Calibration category already exists: {category_root}")
        shutil.rmtree(category_root)

    train_good = category_root / "train" / "good"
    test_good = category_root / "test" / ("normal_calibration" if encode_normals_with_zero_masks else "good")
    test_synthetic = category_root / "test" / "synthetic"
    ground_truth = category_root / "ground_truth" / "synthetic"
    normal_ground_truth = category_root / "ground_truth" / "normal_calibration"
    directories = [train_good, test_good, test_synthetic, ground_truth]
    if encode_normals_with_zero_masks:
        directories.append(normal_ground_truth)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled = source_images.copy()
    rng.shuffle(shuffled)
    selected_normals = shuffled[: min(normal_count, len(shuffled))]
    for index, source in enumerate(selected_normals):
        image_name = f"{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, test_good / image_name)
        if encode_normals_with_zero_masks:
            with Image.open(source) as image:
                zero_mask = Image.new("L", image.size, 0)
            zero_mask.save(normal_ground_truth / f"{index:03d}_mask.png")
    for index, source in enumerate(shuffled[: min(8, len(shuffled))]):
        shutil.copy2(source, train_good / f"{index:03d}{source.suffix.lower()}")

    modes = ("texture", "dark", "bright", "scratch")
    records: list[dict[str, str | int]] = []
    for index in range(anomaly_count):
        target_path = source_images[rng.randrange(len(source_images))]
        donor_path = source_images[rng.randrange(len(source_images))]
        mode = modes[index % len(modes)]
        with Image.open(target_path) as target, Image.open(donor_path) as donor:
            synthetic, mask = synthesize(target, donor, rng, mode)
        image_name = f"{index:03d}.png"
        mask_name = f"{index:03d}_mask.png"
        synthetic.save(test_synthetic / image_name)
        mask.save(ground_truth / mask_name)
        records.append({"image": image_name, "mask": mask_name, "mode": mode, "source": target_path.name})

    metadata = {
        "category": category,
        "seed": seed,
        "anomaly_count": anomaly_count,
        "normal_count": len(selected_normals),
        "normal_encoding": "zero_mask_subclass" if encode_normals_with_zero_masks else "good",
        "source": str(resolved_normal_dir) if resolved_normal_dir is not None else "explicit_train_normal_list",
        "records": records,
    }
    (category_root / "calibration_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return category_root
