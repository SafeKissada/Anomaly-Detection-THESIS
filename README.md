# Project-Thesis: ConvNeXt Feature-Space Autoencoder for Visual Anomaly Detection

Unsupervised visual anomaly detection: a frozen, pretrained **ConvNeXt** backbone extracts multi-stage image features, and a lightweight **convolutional autoencoder** is trained to reconstruct those features using only *normal* images. At inference, the reconstruction error (SSIM + MSE) between real and reconstructed features is used to produce an anomaly score and a spatial heatmap for each image — no anomaly examples are needed for training.

## Description

The pipeline:

1. Scans `DATA_ROOT/good` and `DATA_ROOT/defect`, labelling images `normal`/`anomaly` directly from the folder they're under (no filename-keyword guessing), then computes a seed-based, cached split: **good is split 3-way into train/val/test**; **defect is split 2-way into val/test only** — defect images can never be assigned to train, in training or scoring, by construction (see `scan_and_split()` in `src/data/dataset.py`).
2. Extracts frozen ConvNeXt Stage-2 + Stage-3 features and per-channel normalizes them using statistics fitted on normal images only.
3. Trains a small conv autoencoder (with residual "LightCNBlock" refinement stages) to reconstruct normal-image features, using early stopping on validation AUROC.
4. Scores every image by aggregating the per-pixel SSIM+MSE reconstruction error (mean / max / top-k%) into a single anomaly score, and upsamples the error map into a Gaussian-smoothed heatmap.
5. Picks a deployment threshold from a percentile of validation normal scores (plus an oracle max-F1 threshold for diagnostics only), then reports AUC-ROC, Average Precision, Accuracy, Precision, Recall, and F1 per split.
6. Persists every artifact (history, scores, heatmaps, checkpoints, threshold) to disk so plotting/visualization can be re-run later without retraining.

## Tech Stack

- **Language:** Python 3.13
- **Deep learning:** PyTorch, TorchVision (pretrained ConvNeXt: tiny/small/base/large)
- **CV / image I/O:** Pillow, OpenCV (`opencv-python`)
- **Scientific / ML:** NumPy, SciPy (Gaussian smoothing), scikit-learn (ROC/PR/metrics)
- **Data handling:** pandas
- **Visualization:** Matplotlib, Seaborn
- **CLI output:** rich (console panels/tables), tqdm (progress bars)

## Project Structure

```
config
  ├── config.py              # Config dataclass (paths, hyperparameters, seed) + set_seed()
scripts
  ├── train.py               # Entry point: scan data -> extract features -> train AE -> score -> save artifacts
  ├── visualize.py           # Entry point: loads saved artifacts and renders all plots/heatmaps/gallery
src/
  ├── data/dataset.py        # Directory scanning, filename labeling, AnomalyDataset, transforms, DataLoader factory
  ├── model/
      backbone_baseline.py   # ConvNeXtExtractor: frozen backbone, feature fusion, normalization fitting
      autoencoder.py         # FeatureAutoencoder: conv encoder/decoder with LightCNBlock residual refinement
  ├── engine.py              # Training loop, EarlyStopping, heatmap generation, anomaly scoring
  ├── losses.py              # SSIMLoss, CombinedLoss (SSIM+MSE), loss factory
  ├── evaluate.py            # Metrics computation, percentile threshold, oracle F1 threshold diagnostic
  ├── io_utils.py            # Save/load helpers for all training artifacts
  ├── visual.py              # Plotting functions (EDA, ROC/PR curves, confusion matrices, heatmaps, gallery)
requirements.txt
```

## Installation

Requires Python 3.10+ and, for GPU acceleration, a CUDA-capable GPU with a matching PyTorch build.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Configuration

Before running, edit `config/config.py` and set your dataset path (defaults to a placeholder string):

```python
DATA_ROOT     : str = "path to folder containing good/ and defect/ subfolders"
GOOD_DIRNAME  : str = "good"     # normal images
DEFECT_DIRNAME: str = "defect"   # anomaly images
SPLIT_RATIOS  : Tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test

SAVE_PATH   : str = "save path(log)"       # checkpoints, history, threshold, scores
OUTPUT_PATH : str = "resual path(visual)"  # plots, predictions, gallery
```

`DATA_ROOT` must contain exactly two subfolders, `good/` and `defect/`, each scanned recursively for images (`.jpg`, `.jpeg`, `.png`, `.bmp`). The label comes directly from which folder a file is under — there is no filename-keyword guessing. The split (computed once, then cached to `SPLIT_CACHE_PATH`) is **class-specific**:

- `good` images are split 3-way into **train / val / test** using `SPLIT_RATIOS` as-is.
- `defect` images are split 2-way into **val / test only**, using just the val:test portion of `SPLIT_RATIOS` (renormalized to sum to 1). Defect images can **never** be assigned to train — not for training, and not for scoring — and this is checked in code (a `RuntimeError` is raised if it's ever violated) rather than left as a togglable option.

Only normal (`good`) images from the train split are ever used to train the autoencoder; val and test each contain a mix of good + defect for evaluation. Other tunables (backbone variant, image size, batch size, epochs, loss weights, scoring method, threshold percentile, etc.) also live in this file.

## Usage

Run from the project root so the `config`/`src` packages resolve correctly.

```bash
# 1. Train the autoencoder, score all splits, and save every artifact
python scripts/train.py

# 2. Render plots, ROC/PR curves, heatmaps, and the result gallery from saved artifacts
python scripts/visualize.py
```

- `train.py` writes logs to `logs/train_<timestamp>.log`, checkpoints/history/threshold to `SAVE_PATH`, and per-split score arrays + prediction CSVs + `final_results.json` to `OUTPUT_PATH`.
- `visualize.py` never retrains or re-scores — it only reads the artifacts `train.py` produced (it will raise a clear error if any are missing).
