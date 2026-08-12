# Anomaly Detection THESIS

Visual anomaly / defect detection using a **frozen ConvNeXt feature extractor** + a **trainable convolutional autoencoder** on the extracted feature maps. The autoencoder learns to reconstruct features of *normal (good)* images only; at inference time, reconstruction error is turned into a pixel-level heatmap and an image-level anomaly score, which is thresholded to flag defects.

## How it works

1. **Feature extraction** — A pretrained, frozen `torchvision` ConvNeXt (`tiny` / `small` / `base` / `large`) extracts multi-stage features (Stage2 + Stage3, concatenated) from each image. Per-channel mean/std normalization statistics are fitted once on the *normal* training set only.
2. **Autoencoder** — A lightweight convolutional encoder–decoder (ConvNeXt-style blocks with depthwise conv + residual connections) compresses the normalized features to a bottleneck and reconstructs them. It is trained **only on normal (good) images**.
3. **Anomaly scoring** — At inference, the per-pixel reconstruction error (MSE / MAE / Huber / SSIM / SSIM+MSE, configurable) is upsampled, Gaussian-smoothed into a heatmap, and aggregated into a single image-level score (`mean` / `max` / `top-k%`).
4. **Thresholding** — A deployment threshold is chosen as a percentile of the validation normal-image scores. An oracle (max-F1) threshold is also computed as a diagnostic only — it is **not** used for reported metrics, since it uses validation anomaly labels.
5. **Evaluation & visualization** — AUC-ROC, average precision, accuracy, precision, recall, F1 are computed on val/test; ROC/PR curves, confusion matrices, score distributions, training curves, and image/heatmap galleries are rendered separately.

## Project structure

```
.
├── RUN.py                     # One-shot entry point: patches Config defaults, then runs train → visualize
├── config/
│   └── config.py               # Dataclass Config with every hyperparameter + validation in __post_init__
├── scripts/
│   ├── train.py                 # Train autoencoder → score val/test → save all artifacts
│   ├── visualize.py             # Load saved artifacts → render all plots/heatmaps/galleries
│   └── ssim_check.py            # Sanity check for SSIM loss on z-score-normalized features before a full run
├── src/
│   ├── data/
│   │   └── dataset.py            # Dataset scan/split, color-mode transforms, Dataset/DataLoader builders
│   ├── model/
│   │   ├── backbone_baseline.py   # Frozen ConvNeXt feature extractor (Stage2+Stage3)
│   │   └── autoencoder.py         # Feature-space convolutional autoencoder
│   ├── engine.py                 # Training loop, EarlyStopping, scoring/heatmap pipeline
│   ├── losses.py                 # MSE / MAE / Huber / SSIM / SSIM+MSE loss + get_criterion()
│   ├── optimes.py                # Optimizer factory (Adam / AdamW / SGD / RMSprop) driven by cfg.OPTIM
│   ├── evaluate.py               # Metrics, percentile threshold, oracle threshold diagnostic
│   ├── io_utils.py               # Save/load history, scores, checkpoints, norm stats, threshold
│   └── visual.py                 # All plotting/heatmap/gallery functions
└── requirements.txt
```

## Requirements

- Python 3.10+
- A CUDA GPU is recommended but not required (`Config.DEVICE` falls back to CPU automatically)

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages: `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, `rich`.

## Dataset layout

Point `Config.DATA_ROOT` at a folder containing two subfolders (names configurable via `GOOD_DIRNAME` / `DEFECT_DIRNAME`):

```
<DATA_ROOT>/
├── all_good/      # normal images (used to train the autoencoder)
└── all_defect/    # defective / anomalous images
```

Supported extensions default to `.jpg .jpeg .png .bmp` (`Config.VALID_EXT`). Images are split 70/15/15 into train/val/test by default (`Config.SPLIT_RATIOS`); the split assignment is cached to `Config.SPLIT_CACHE_PATH` so re-runs are reproducible. **Only "good" images ever land in the training set** — this is asserted at split time.

## Usage

### Quick start

`RUN.py` overrides `Config` defaults (dataset paths, hyperparameters, etc.) without editing `config/config.py`, then runs training followed by visualization:

```bash
python RUN.py
```

Edit the `OVERRIDES` dict at the top of `RUN.py` (or `config/config.py` directly) to point at your dataset and adjust hyperparameters.

### Run steps individually

```bash
# 1. (Optional) sanity-check the SSIM loss before a full run
python scripts/ssim_check.py

# 2. Train the autoencoder, score val/test, save all artifacts
python scripts/train.py

# 3. Render plots, heatmaps, and image galleries from the saved artifacts
python scripts/visualize.py
```

`scripts/visualize.py` only *reads* artifacts saved by `scripts/train.py` (history, scores, checkpoint, threshold, norm stats) — it never retrains or re-scores. If artifacts are missing, it raises an error telling you to run `train.py` first.

## Key configuration options

All options live in `config/config.py` (`Config` dataclass); see the file for full documentation on each field. Highlights:

| Setting | Options |
|---|---|
| `BACKBONE` | `tiny` \| `small` \| `base` \| `large` (ConvNeXt variant) |
| `LOSS` | `MSE` \| `MAE`/`L1` \| `HUBER`/`SMOOTH_L1` \| `SSIM` \| `SSIM_MSE` |
| `OPTIM` | `Adam` \| `AdamW` \| `SGD` \| `RMSprop` |
| `SCORE_METHOD` | `mean` \| `max` \| `topk` (image-level score aggregation) |
| Color mode | `RGB` (default) \| `GRAYSCALE` \| `GRAYSCALE_EQUALIZATION` \| `GRAYSCALE_CLAHE` \| `GRAYSCALE_EQUALIZATION_CLAHE` — controlled via `USE_GRAYSCALE`, `USE_GRAYSCALE_EQUALIZATION`, `USE_CLAHE` |

`Config.__post_init__` validates `OPTIM`, `SPLIT_RATIOS` (must sum to 1.0, exactly 3 values), and fails fast with an actionable error if `DATA_ROOT` is left as the placeholder or doesn't exist.

## Outputs

Artifacts are written under `Config.SAVE_PATH` (checkpoints/logs) and `Config.OUTPUT_PATH` (results), including:

- `extractor_norm_stats.pt` — fitted feature normalization stats
- `history.json` — per-epoch training/validation loss & AUROC
- `scores_val.npz`, `scores_test.npz` — per-image scores, labels, heatmaps
- `threshold.json` — deployment threshold + oracle diagnostic
- `predictions_val.csv`, `predictions_test.csv` — per-image predictions
- `final_results.json` — full run summary (config snapshot + metrics), for comparing experiments
- Trained autoencoder checkpoint (`.pth`)

Console output also prints a live-updating results table (AUC-ROC, Average Precision, Accuracy, Precision, Recall, F1) for validation and test splits.