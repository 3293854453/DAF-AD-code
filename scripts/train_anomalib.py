"""Train one anomalib branch for one dataset category and seed."""

from __future__ import annotations

import argparse
import os

# This container advertises a very high CPU count, so PyTorch defaults to dozens
# of intra-op threads. For single-image transforms that causes severe thread
# oversubscription (measured ~580 ms/image instead of ~4 ms/image). Cap the
# thread pools before importing torch.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import random
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", choices=("efficient_ad", "patchcore"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", default="/tmp/dafad-runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--imagenet_dir", default="datasets/imagenette/train")
    parser.add_argument("--max_steps", type=int, default=70000)
    parser.add_argument("--local_backbone_weights")
    parser.add_argument(
        "--train_normal_dir",
        help="Optional disjoint detector-training normal directory. When set, use Folder instead of the benchmark datamodule.",
    )
    parser.add_argument("--normal_test_dir")
    parser.add_argument("--abnormal_dir")
    parser.add_argument("--mask_dir")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from anomalib.data import Folder, MVTecAD, Visa
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Patchcore

    custom_dirs = (args.train_normal_dir, args.normal_test_dir, args.abnormal_dir, args.mask_dir)
    if any(custom_dirs) and not all(custom_dirs):
        raise ValueError(
            "--train_normal_dir, --normal_test_dir, --abnormal_dir, and --mask_dir "
            "must be supplied together"
        )
    if all(custom_dirs):
        datamodule = Folder(
            name=f"{args.dataset}_{args.category}_independent_train",
            normal_dir=args.train_normal_dir,
            normal_test_dir=args.normal_test_dir,
            abnormal_dir=args.abnormal_dir,
            mask_dir=args.mask_dir,
            train_batch_size=args.train_batch,
            eval_batch_size=args.eval_batch,
            num_workers=args.workers,
            val_split_mode="same_as_test",
            seed=args.seed,
        )
        print(
            "independent protocol: detector training uses only "
            f"{Path(args.train_normal_dir).resolve()}"
        )
    else:
        data_class = MVTecAD if args.dataset == "mvtec_ad" else Visa
        datamodule = data_class(
            root=args.root,
            category=args.category,
            train_batch_size=args.train_batch,
            eval_batch_size=args.eval_batch,
            num_workers=args.workers,
            seed=args.seed,
        )
    if args.model == "efficient_ad":
        if args.train_batch != 1:
            raise ValueError("EfficientAD requires --train_batch 1")
        model = EfficientAd(model_size="small", imagenet_dir=args.imagenet_dir)
    else:
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            pre_trained=args.local_backbone_weights is None,
        )
        if args.local_backbone_weights:
            from safetensors.torch import load_file

            backbone = model.model.feature_extractor.feature_extractor
            incompatible = backbone.load_state_dict(load_file(args.local_backbone_weights), strict=False)
            if incompatible.missing_keys:
                raise ValueError(f"Local backbone weights are incompatible: {incompatible}")
            if incompatible.unexpected_keys:
                print(
                    "ignored unused backbone keys: "
                    f"{len(incompatible.unexpected_keys)} (expected for truncated layer2/layer3 extraction)"
                )
            print(f"local backbone weights loaded: {args.local_backbone_weights}")
    output = Path(args.output) / args.dataset / args.category / args.model / f"seed{args.seed}"
    engine_options = dict(
        default_root_dir=output,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    if args.model == "efficient_ad":
        engine_options["max_steps"] = args.max_steps
    engine = Engine(**engine_options)
    engine.fit(model=model, datamodule=datamodule)
    print(f"training complete: {output.resolve()}")


if __name__ == "__main__":
    main()
