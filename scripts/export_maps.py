"""Export aligned anomaly maps and masks from one anomalib checkpoint."""

from __future__ import annotations

import argparse
import os

# See scripts/train_anomalib.py: avoid thread oversubscription on this host.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_num_threads(1)


def _field(batch: Any, name: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _as_maps(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Prediction batch does not contain {name}")
    array = torch.as_tensor(value).detach().cpu().numpy()
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"Expected BxHxW {name}, got {array.shape}")
    return array


def _as_paths(value: Any, count: int, offset: int) -> list[str]:
    if value is None:
        return [f"sample_{offset + index:06d}" for index in range(count)]
    if isinstance(value, (str, Path)):
        value = [value]
    paths = [str(path) for path in value]
    if len(paths) != count:
        raise ValueError(f"Expected {count} image paths, got {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("mvtec_ad", "visa", "folder_calibration"),
        default="mvtec_ad",
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", choices=("efficient_ad", "patchcore"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imagenet_dir", default="datasets/imagenette/train")
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="gpu")
    parser.add_argument(
        "--val_split_mode",
        choices=("none", "same_as_test", "from_train", "from_test", "synthetic", "from_dir"),
        default="same_as_test",
    )
    args = parser.parse_args()

    from anomalib.data import Folder, MVTecAD, Visa
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Patchcore

    if args.dataset == "folder_calibration":
        category_root = Path(args.dataset_root).resolve() / args.category
        datamodule = Folder(
            name=f"calibration_{args.category}",
            normal_dir=category_root / "train" / "good",
            normal_test_dir=category_root / "test" / "good",
            abnormal_dir=category_root / "test" / "synthetic",
            mask_dir=category_root / "ground_truth" / "synthetic",
            train_batch_size=1,
            eval_batch_size=args.eval_batch,
            num_workers=args.workers,
            val_split_mode=args.val_split_mode,
            seed=0,
        )
    else:
        data_class = MVTecAD if args.dataset == "mvtec_ad" else Visa
        datamodule = data_class(
            root=args.dataset_root,
            category=args.category,
            train_batch_size=1,
            eval_batch_size=args.eval_batch,
            num_workers=args.workers,
            val_split_mode=args.val_split_mode,
            seed=0,
        )
    if args.model == "efficient_ad":
        model = EfficientAd(model_size="small", imagenet_dir=args.imagenet_dir)
    else:
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            pre_trained=False,
        )
    engine = Engine(
        default_root_dir=f"/tmp/dafad-export/{args.category}/{args.model}",
        accelerator=args.device if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.checkpoint,
        return_predictions=True,
    )
    if predictions is None:
        raise RuntimeError("anomalib returned no predictions")

    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    paths: list[str] = []
    offset = 0
    for batch in predictions:
        anomaly_maps = _as_maps(_field(batch, "anomaly_map"), "anomaly_map").astype(np.float32)
        gt_masks = _as_maps(_field(batch, "gt_mask"), "gt_mask").astype(np.uint8)
        if anomaly_maps.shape != gt_masks.shape:
            raise ValueError(f"Map/mask shape mismatch: {anomaly_maps.shape} vs {gt_masks.shape}")
        batch_paths = _as_paths(_field(batch, "image_path"), len(anomaly_maps), offset)
        maps.append(anomaly_maps)
        masks.append(gt_masks)
        paths.extend(batch_paths)
        offset += len(anomaly_maps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        maps=np.concatenate(maps),
        masks=(np.concatenate(masks) > 0).astype(np.uint8),
        paths=np.asarray(paths),
        model=np.asarray(args.model),
    )
    print(f"saved {len(paths)} maps: {output.resolve()}")


if __name__ == "__main__":
    main()
