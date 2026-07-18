# Project-Thesis: ConvNeXt Feature-Space Autoencoder for Visual Anomaly Detection

Unsupervised visual anomaly detection: a frozen, pretrained **ConvNeXt** backbone extracts multi-stage image features, and a lightweight **convolutional autoencoder** is trained to reconstruct those features using only *normal* images. At inference, the reconstruction error (SSIM + MSE) between real and reconstructed features is used to produce an anomaly score and a spatial heatmap for each image — no anomaly examples are needed for training.

## Description

The pipeline:

1. Scans train/val/test image directories and weakly labels each image as `normal` or `anomaly` from filename keywords (e.g. `good`, `false_call` vs. `defect`, `ng`).
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

Before running, edit `config/config.py` and set your dataset paths (they default to placeholder strings):

```python
TRAIN_DIR   : str = "train dataset path"
VAL_DIR     : str = "validation dataset path"
TEST_DIR    : str = "test dataset path"
SAVE_PATH   : str = "save path(log)"       # checkpoints, history, threshold, scores
OUTPUT_PATH : str = "resual path(visual)"  # plots, predictions, gallery
```

Each directory is scanned recursively for images (`.jpg`, `.jpeg`, `.png`, `.bmp`); labels are inferred from filenames via `NORMAL_KEYWORDS` / `ANOMALY_KEYWORDS`. Other tunables (backbone variant, image size, batch size, epochs, loss weights, scoring method, threshold percentile, etc.) also live in this file.

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
