"""Run the frozen DAF-AD category pipeline with resume support."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def execute(arguments: list[str], dry_run: bool, offline: bool = False) -> None:
    print("+", shlex.join(arguments), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def latest_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.rglob("model.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model.ckpt found under {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), default="mvtec_ad")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--imagenet_dir", required=True)
    parser.add_argument("--local_backbone_weights", required=True)
    parser.add_argument("--workspace", default="/tmp/dafad-batch")
    parser.add_argument("--artifact_root", default="/mnt/DAF-AD")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patch_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=70000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument("--unlock_token", default="FROZEN_DAFAD_V1")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    dataset_root = Path(args.dataset_root).resolve()
    workspace = Path(args.workspace).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    checkpoint_dir = artifact_root / "checkpoints"
    calibration_dir = artifact_root / "calibration"
    result_dir = artifact_root / "results"
    if args.dataset != "mvtec_ad":
        checkpoint_dir = checkpoint_dir / args.dataset
        calibration_dir = calibration_dir / args.dataset
        result_dir = result_dir / args.dataset
    dataset_calibration = calibration_dir / f"{args.dataset}_dataset_seed{args.seed}.json"
    for directory in (workspace, checkpoint_dir, calibration_dir, result_dir):
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        print(f"\n=== {category} ===", flush=True)
        normal_dir = dataset_root / category / "train" / "good"
        if args.dataset == "mvtec_ad":
            if not args.dry_run and not normal_dir.is_dir():
                raise FileNotFoundError(f"Missing normal training directory: {normal_dir}")
        else:
            split_csv = dataset_root / "split_csv" / "1cls.csv"
            if not args.dry_run and not split_csv.is_file():
                raise FileNotFoundError(f"Missing VisA split CSV: {split_csv}")

        checkpoints: dict[str, Path] = {}
        for model in ("patchcore", "efficient_ad"):
            target = checkpoint_dir / f"{category}_{model}_seed{args.seed}.ckpt"
            checkpoints[model] = target
            if target.exists():
                print(f"resume: checkpoint exists: {target}", flush=True)
                continue
            command = [
                python,
                "scripts/train_anomalib.py",
                "--dataset",
                args.dataset,
                "--root",
                str(dataset_root),
                "--category",
                category,
                "--model",
                model,
                "--seed",
                str(args.seed),
                "--eval_batch",
                str(args.eval_batch),
                "--workers",
                str(args.workers),
                "--output",
                str(workspace / "runs"),
            ]
            if model == "patchcore":
                command += [
                    "--train_batch",
                    str(args.patch_batch),
                    "--local_backbone_weights",
                    args.local_backbone_weights,
                ]
            else:
                command += [
                    "--train_batch",
                    "1",
                    "--imagenet_dir",
                    args.imagenet_dir,
                    "--max_steps",
                    str(args.max_steps),
                ]
            execute(command, args.dry_run, offline=model == "patchcore")
            if not args.dry_run:
                run_directory = workspace / "runs" / args.dataset / category / model / f"seed{args.seed}"
                shutil.copy2(latest_checkpoint(run_directory), target)
                print(f"saved checkpoint: {target}", flush=True)

        synthetic_root = workspace / "synthetic"
        calibration_json = calibration_dir / f"{category}_seed{args.seed}.json"
        maps_dir = workspace / "maps"
        if not args.dry_run:
            maps_dir.mkdir(parents=True, exist_ok=True)
        cal_maps: dict[str, Path] = {}

        calibration_is_current = False
        if calibration_json.exists():
            try:
                existing = json.loads(calibration_json.read_text(encoding="utf-8"))
                calibration_is_current = all(
                    key in existing for key in ("adaptive", "profile", "fixed", "branches", "mad_mean")
                )
            except (OSError, json.JSONDecodeError):
                calibration_is_current = False
        if not calibration_is_current:
            generation_command = [
                    python,
                    "scripts/generate_calibration.py",
                    "--output_root",
                    str(synthetic_root),
                    "--category",
                    category,
                    "--anomaly_count",
                    "32",
                    "--normal_count",
                    "16",
                    "--seed",
                    str(args.seed),
                    "--overwrite",
                ]
            if args.dataset == "mvtec_ad":
                generation_command += ["--normal_dir", str(normal_dir)]
            else:
                generation_command += ["--visa_root", str(dataset_root)]
            execute(generation_command, args.dry_run)
            for model in ("efficient_ad", "patchcore"):
                output = maps_dir / f"{category}_cal_{model}.npz"
                cal_maps[model] = output
                command = [
                    python,
                    "scripts/export_maps.py",
                    "--dataset",
                    "folder_calibration",
                    "--dataset_root",
                    str(synthetic_root),
                    "--category",
                    category,
                    "--model",
                    model,
                    "--checkpoint",
                    str(checkpoints[model]),
                    "--output",
                    str(output),
                    "--eval_batch",
                    str(args.eval_batch),
                    "--workers",
                    str(args.workers),
                    "--val_split_mode",
                    "none",
                ]
                if model == "efficient_ad":
                    command += ["--imagenet_dir", args.imagenet_dir]
                execute(command, args.dry_run, offline=model == "patchcore")
            merged = maps_dir / f"{category}_cal_merged.npz"
            execute(
                [
                    python,
                    "scripts/merge_maps.py",
                    "--efficient",
                    str(cal_maps["efficient_ad"]),
                    "--patchcore",
                    str(cal_maps["patchcore"]),
                    "--output",
                    str(merged),
                ],
                args.dry_run,
            )
            execute(
                [
                    python,
                    "scripts/calibrate_fusion.py",
                    "--maps",
                    str(merged),
                    "--output",
                    str(calibration_json),
                ],
                args.dry_run,
            )
        else:
            print(f"resume: calibration exists: {calibration_json}", flush=True)

    calibration_arguments = [
        f"{category}={calibration_dir / f'{category}_seed{args.seed}.json'}"
        for category in args.categories
    ]
    execute(
        [
            python,
            "scripts/calibrate_dataset.py",
            "--calibrations",
            *calibration_arguments,
            "--output",
            str(dataset_calibration),
        ],
        args.dry_run,
    )
    if not args.evaluate_test:
        print("test remains locked; rerun with --evaluate_test after dataset calibration is frozen", flush=True)
        return

    for category in args.categories:
        print(f"\n=== Test evaluation: {category} ===", flush=True)
        checkpoints = {
            model: checkpoint_dir / f"{category}_{model}_seed{args.seed}.ckpt"
            for model in ("patchcore", "efficient_ad")
        }
        maps_dir = workspace / "maps"
        if not args.dry_run:
            maps_dir.mkdir(parents=True, exist_ok=True)
        test_maps: dict[str, Path] = {}
        for model in ("efficient_ad", "patchcore"):
            output = maps_dir / f"{category}_test_{model}.npz"
            test_maps[model] = output
            if output.exists():
                print(f"resume: test maps exist: {output}", flush=True)
                continue
            command = [
                python,
                "scripts/export_maps.py",
                "--dataset",
                args.dataset,
                "--dataset_root",
                str(dataset_root),
                "--category",
                category,
                "--model",
                model,
                "--checkpoint",
                str(checkpoints[model]),
                "--output",
                str(output),
                "--eval_batch",
                str(args.eval_batch),
                "--workers",
                str(args.workers),
            ]
            if model == "efficient_ad":
                command += ["--imagenet_dir", args.imagenet_dir]
            execute(command, args.dry_run, offline=model == "patchcore")
        merged_test = maps_dir / f"{category}_test_merged.npz"
        execute(
            [
                python,
                "scripts/merge_maps.py",
                "--efficient",
                str(test_maps["efficient_ad"]),
                "--patchcore",
                str(test_maps["patchcore"]),
                "--output",
                str(merged_test),
            ],
            args.dry_run,
        )
        execute(
            [
                python,
                "scripts/evaluate_fusion.py",
                "--maps",
                str(merged_test),
                "--calibration",
                str(dataset_calibration),
                "--category",
                category,
                "--unlock_token",
                args.unlock_token,
                "--output",
                str(result_dir / f"{category}_fusion_seed{args.seed}.json"),
            ],
            args.dry_run,
        )
        print(f"completed category: {category}", flush=True)

    execute(
        [
            python,
            "scripts/summarize_results.py",
            "--results",
            *[str(result_dir / f"{category}_fusion_seed{args.seed}.json") for category in args.categories],
            "--output",
            str(result_dir / f"{args.dataset}_summary_seed{args.seed}.json"),
        ],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
