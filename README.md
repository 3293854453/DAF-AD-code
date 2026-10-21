# DAF-AD code and reproducibility scripts

Code accompanying the manuscript **Training-side synthetic calibration for dual-branch industrial anomaly map fusion: A granularity study**.

This release contains the manuscript-associated workflow only.

## Contents

- `src/dafad/`: synthetic-defect generation, map normalization and fusion, ensemble baselines, and metric implementations.
- `scripts/`: detector training, calibration generation, anomaly-map export and merging, per-category and dataset-global calibration, evaluation, and macro-summary generation.
- `requirements.txt`: Python dependencies. Fusion calibration defaults are provided as command-line options in `scripts/calibrate_fusion.py`; dataset paths and categories are supplied when launching `scripts/run_pipeline.py`.

No benchmark images or masks, detector checkpoints, pretrained weights, calibration outputs, result JSON files, or paper figures are redistributed here. Obtain datasets from their official distribution channels and comply with their terms. MVTec AD: <https://www.mvtec.com/company/research/datasets/mvtec-ad>. VisA: <https://github.com/amazon-science/spot-diff>.

## Environment and usage

The manuscript records Python 3.12.4, PyTorch 2.4.0+cu121, torchvision 0.19.0+cu121, Anomalib 2.6.0, and cuDNN 9.1.0. Install the dependencies in `requirements.txt` in a compatible CUDA/PyTorch environment. Dataset roots, ImageNette files used by EfficientAD, and PatchCore initialization weights must be obtained separately and are intentionally excluded.

Workflow scripts expose command-line arguments; run them with `--help` before use. The pipeline first calibrates each requested category, selects a dataset-global weight by macro-average calibration AUROC (macro-average best F1 breaks ties), then evaluates fixed fusion, dataset-global fusion, category-adaptive fusion, MAD-mean, and both detector branches. It writes per-category metrics and an unweighted dataset macro-average summary. Supply every benchmark category to reproduce a full-dataset macro average. For example:

```bash
python scripts/run_pipeline.py \
  --dataset mvtec_ad \
  --dataset_root /path/to/mvtec_ad \
  --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
  --imagenet_dir /path/to/imagenette/train \
  --local_backbone_weights /path/to/wide_resnet50_2.pth \
  --evaluate_test --unlock_token FROZEN_DAFAD_V1
```

Omit `--evaluate_test` to stop after training-side calibration and weight selection. AUPRO is computed with Anomalib 2.6.0 at an FPR limit of 0.30.

## Data and model assets

MVTec AD and VisA images/masks remain subject to their respective dataset terms and are not included. ImageNette/ImageNet assets, model checkpoints, and the PatchCore initialization used in the recorded run are not included.

## Citation and reuse

Please cite the associated manuscript when using this code. Large model checkpoints and other assets remain available from the authors upon reasonable request, subject to dataset and model terms.

No open-source license has been specified for this code. This repository is public for inspection by editors, reviewers, and readers. Unless the authors add a license, no permission is granted to reuse, modify, or redistribute the code. Dataset and pretrained-weight terms are separate and remain applicable.
