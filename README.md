# Anomaly Detection THESI

## English

Visual anomaly / defect detection using a **frozen ConvNeXt feature extractor** + a **trainable convolutional autoencoder** on the extracted feature maps. The autoencoder learns to reconstruct features of *normal (good)* images only; at inference time, reconstruction error is turned into a pixel-level heatmap and an image-level anomaly score, which is thresholded to flag defects.

### How it works

1. **Feature extraction** — A pretrained, frozen `torchvision` ConvNeXt (`tiny` / `small` / `base` / `large`) extracts multi-stage features (Stage2 + Stage3, concatenated) from each image. Per-channel mean/std normalization statistics are fitted once on the *normal* training set only.
2. **Autoencoder** — A lightweight convolutional encoder–decoder (ConvNeXt-style blocks with depthwise conv + residual connections) compresses the normalized features to a bottleneck and reconstructs them. It is trained **only on normal (good) images**.
3. **Anomaly scoring** — At inference, the per-pixel reconstruction error (`MSE` / `MAE` / `HUBER` / `COS` / `COS_MSE`, configurable) is upsampled, Gaussian-smoothed into a heatmap, and aggregated into a single image-level score (`mean` / `max` / `top-k%`).
4. **Thresholding** — A deployment threshold is chosen as a percentile of the validation normal-image scores. An oracle (max-F1) threshold is also computed as a diagnostic only — it is **not** used for reported metrics, since it uses validation anomaly labels.
5. **Evaluation & visualization** — AUC-ROC, average precision, accuracy, precision, recall, F1 are computed on val/test; ROC/PR curves, confusion matrices, score distributions, training curves, and image/heatmap galleries are rendered separately.

### Project structure

```
.
├── RUN.py                     # One-shot entry point: patches Config defaults, then runs train → visualize
├── config/
│   └── config.py               # Dataclass Config with every hyperparameter + validation in __post_init__
├── scripts/
│   ├── train.py                 # Train autoencoder → score val/test → save all artifacts
│   └── visualize.py             # Load saved artifacts → render all plots/heatmaps/galleries
├── src/
│   ├── data/
│   │   └── dataset.py            # Dataset scan/split, color-mode transforms, Dataset/DataLoader builders
│   ├── model/
│   │   ├── backbone_baseline.py   # Frozen ConvNeXt feature extractor (Stage2+Stage3)
│   │   └── autoencoder.py         # Feature-space convolutional autoencoder
│   ├── engine.py                 # Training loop, EarlyStopping, scoring/heatmap pipeline
│   ├── losses.py                 # MSE / MAE / Huber / Cosine / Cosine+MSE loss + get_criterion()
│   ├── optimes.py                # Optimizer factory (Adam / AdamW / SGD / RMSprop) driven by cfg.OPTIM
│   ├── evaluate.py               # Metrics, percentile threshold, oracle threshold diagnostic
│   ├── io_utils.py               # Save/load history, scores, checkpoints, norm stats, threshold
│   └── visual.py                 # All plotting/heatmap/gallery functions
└── requirements.txt
```

### Requirements

- Python 3.10+
- A CUDA GPU is recommended but not required (`Config.DEVICE` falls back to CPU automatically)

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages: `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, `rich`.


### Dataset layout

Point `Config.DATA_ROOT` at a folder containing two subfolders (names configurable via `GOOD_DIRNAME` / `DEFECT_DIRNAME`):

```
<DATA_ROOT>/
├── good/      # normal images (used to train the autoencoder)
└── defect/    # defective / anomalous images
```

Supported extensions default to `.jpg .jpeg .png .bmp` (`Config.VALID_EXT`). Images are split 70/15/15 into train/val/test by default (`Config.SPLIT_RATIOS`); the split assignment is cached to `Config.SPLIT_CACHE_PATH` so re-runs are reproducible. **Only "good" images ever land in the training set** — this is asserted at split time.

### Usage

#### Quick start

`RUN.py` overrides `Config` defaults (dataset paths, hyperparameters, etc.) without editing `config/config.py`, then runs training followed by visualization:

```bash
python RUN.py
```

Edit the `OVERRIDES` dict at the top of `RUN.py` (or `config/config.py` directly) to point at your dataset and adjust hyperparameters — every loss/model/training hyperparameter is exposed there, so ablations never require editing `src/` files.

#### Run steps individually

```bash
# 1. Train the autoencoder, score val/test, save all artifacts
python scripts/train.py

# 2. Render plots, heatmaps, and image galleries from the saved artifacts
python scripts/visualize.py
```

`scripts/visualize.py` only *reads* artifacts saved by `scripts/train.py` (history, scores, checkpoint, threshold, norm stats) — it never retrains or re-scores. If artifacts are missing, it raises an error telling you to run `train.py` first.

### Key configuration options

All options live in `config/config.py` (`Config` dataclass); see the file for full documentation on each field. Highlights:

| Setting | Options |
|---|---|
| `BACKBONE` | `tiny` \| `small` \| `base` \| `large` (ConvNeXt variant) |
| `LOSS` | `MSE` \| `MAE`/`L1` \| `HUBER`/`SMOOTH_L1` \| `COS` \| `COS_MSE` |
| `OPTIM` | `Adam` \| `AdamW` \| `SGD` \| `RMSprop` |
| `SCORE_METHOD` | `mean` \| `max` \| `topk` (image-level score aggregation) |
| Color mode | `RGB` (default) \| `GRAYSCALE` \| `GRAYSCALE_EQUALIZATION` \| `GRAYSCALE_CLAHE` \| `GRAYSCALE_EQUALIZATION_CLAHE` — controlled via `USE_GRAYSCALE`, `USE_GRAYSCALE_EQUALIZATION`, `USE_CLAHE` |

#### Loss functions

The autoencoder is trained on feature maps (per-channel z-score normalized ConvNeXt activations), not raw pixels. `COS`/`COS_MSE` were added specifically because pixel-space losses like SSIM carry assumptions (luminance term, calibrated `data_range`, a window size tuned to image resolution) that don't transfer cleanly to normalized feature space — see `loss_functions_summary.md` for the full rationale (SSIM/SSIM_MSE were removed from this codebase for that reason). Available options:

| `LOSS` value | Config knobs it reads | Notes |
|---|---|---|
| `MSE` | — | Default. Standard per-element squared error. |
| `MAE` / `L1` | — | More robust to outliers than MSE, constant gradient magnitude. |
| `HUBER` / `SMOOTH_L1` | `HUBER_DELTA` | Quadratic below `HUBER_DELTA`, linear above — a reasonable fit here since `HUBER_DELTA` has a direct physical meaning (number of SDs) on z-scored features. |
| `COS` | `COS_EPS` | Per-pixel cosine distance across the channel axis — treats each spatial position's C-length feature vector as a direction. Needs no `data_range`/luminance assumption, so it's stationary across training regardless of how the normalized feature values drift. Scale-invariant by construction: alone, it never penalizes drift in reconstruction magnitude, only direction. |
| `COS_MSE` | `COS_LAM`, `COS_EPS` | Weighted sum of cosine distance (direction / cross-channel activation pattern) and MSE (magnitude). `COS_LAM=1.0` reduces to pure `COS`, `COS_LAM=0.0` reduces to pure `MSE`. Recommended default for feature-space reconstruction — see `loss_functions_summary.md` for the full comparison and rationale. |

`Config.__post_init__` validates `OPTIM`, `LOSS` (against every alias `get_criterion()` recognizes), and `SPLIT_RATIOS` (must sum to 1.0, exactly 3 values), and fails fast with an actionable error if `DATA_ROOT` is left as the placeholder or doesn't exist.

The same criterion is used for both training gradients (`train_autoencoder`) and the error map that feeds the reported val/test heatmaps and scores (`score_dataset_split` → `elementwise_error_map`) — both call sites route through the single dispatch table in `src/losses.py:elementwise_error_map()`, so they cannot silently diverge.

### Outputs

Artifacts are written under `Config.SAVE_PATH` (checkpoints/logs) and `Config.OUTPUT_PATH` (results), including:

- `extractor_norm_stats.pt` — fitted feature normalization stats
- `history.json` — per-epoch training/validation loss & AUROC
- `scores_val.npz`, `scores_test.npz` — per-image scores, labels, heatmaps
- `threshold.json` — deployment threshold + oracle diagnostic
- `predictions_val.csv`, `predictions_test.csv` — per-image predictions
- `final_results.json` — full run summary (a complete `Config` field snapshot via `dataclasses.asdict()`, plus metrics), for comparing experiments without cross-referencing `config.py`
- Trained autoencoder checkpoint (`.pth`)

Console output also prints a live-updating results table (AUC-ROC, Average Precision, Accuracy, Precision, Recall, F1) for validation and test splits.



🇹🇭 [ภาษาไทย](#ภาษาไทย) | 🇬🇧 [English](#english)


---


## ภาษาไทย


ตรวจจับความผิดปกติ/ตำหนิเชิงภาพ (visual anomaly / defect detection) ด้วย **ตัวสกัด feature ConvNeXt แบบ frozen** ร่วมกับ **convolutional 
autoencoder ที่เทรนได้** บน feature map ที่สกัดออกมา autoencoder เรียนรู้ที่จะ reconstruct feature ของภาพ *ปกติ (good)* เท่านั้น เมื่อถึงตอน 
inference reconstruction error จะถูกแปลงเป็น heatmap ระดับพิกเซลและ anomaly score ระดับภาพ ซึ่งจะถูกนำไป threshold เพื่อตัดสินว่าเป็นตำหนิหรือไม่


### วิธีการทำงาน


1. **สกัด feature** — ConvNeXt (`tiny` / `small` / `base` / `large`) ที่ pretrain มาแล้วและ frozen สกัด feature หลาย stage (Stage2 + 
Stage3 นำมาต่อกัน) จากแต่ละภาพ ค่าสถิติ mean/std ต่อ channel สำหรับ normalize ถูก fit เพียงครั้งเดียวจาก training set ที่เป็นภาพ *ปกติ* เท่านั้น
2. **Autoencoder** — encoder-decoder แบบ convolutional ที่เบา (ConvNeXt-style block พร้อม depthwise conv และ residual connection) บีบ
อัด feature ที่ normalize แล้วลงไปที่ bottleneck แล้ว reconstruct กลับ เทรน**เฉพาะภาพปกติ (good)** เท่านั้น
3. **คำนวณ anomaly score** — ตอน inference reconstruction error ต่อพิกเซล (`MSE` / `MAE` / `HUBER` / `COS` / `COS_MSE`, ปรับได้) จะ
ถูก upsample, Gaussian-smooth เป็น heatmap แล้วรวมเป็น score เดียวต่อภาพ (`mean` / `max` / `top-k%`)
4. **ตั้ง threshold** — threshold สำหรับ deployment เลือกจาก percentile ของ score ภาพปกติใน validation set ส่วน oracle threshold (max-
F1) คำนวณไว้เป็น diagnostic เท่านั้น — **ไม่ได้ใช้** ในตัวเลขที่รายงานจริง เพราะใช้ label ของ validation set ซึ่งไม่ควรรู้ตอน deploy จริง
5. **ประเมินผลและ visualize** — คำนวณ AUC-ROC, average precision, accuracy, precision, recall, F1 บน val/test แยกออกมา render 
กราฟ ROC/PR, confusion matrix, score distribution, training curve, และ gallery ภาพ/heatmap ต่างหาก


### โครงสร้างโปรเจกต์


```
.
├── RUN.py                     # entry point เดียวจบ: override ค่า Config แล้วรัน train → visualize
├── config/
│   └── config.py               # dataclass Config รวมทุก hyperparameter + validation ใน __post_init__
├── scripts/
│   ├── train.py                 # เทรน autoencoder → คำนวณ score val/test → เซฟ artifact ทั้งหมด
│   └── visualize.py             # โหลด artifact ที่เซฟไว้ → render กราฟ/heatmap/gallery ทั้งหมด
├── src/
│   ├── data/
│   │   └── dataset.py            # สแกน/แบ่ง dataset, transform ตามโหมดสี, สร้าง Dataset/DataLoader
│   ├── model/
│   │   ├── backbone_baseline.py   # ตัวสกัด feature ConvNeXt แบบ frozen (Stage2+Stage3)
│   │   └── autoencoder.py         # convolutional autoencoder บน feature space
│   ├── engine.py                 # training loop, EarlyStopping, scoring/heatmap pipeline
│   ├── losses.py                 # MSE / MAE / Huber / Cosine / Cosine+MSE loss + get_criterion()
│   ├── optimes.py                # optimizer factory (Adam / AdamW / SGD / RMSprop) ตาม cfg.OPTIM
│   ├── evaluate.py               # metrics, percentile threshold, oracle threshold diagnostic
│   ├── io_utils.py               # save/load history, scores, checkpoint, norm stats, threshold
│   └── visual.py                 # ฟังก์ชัน plot/heatmap/gallery ทั้งหมด
└── requirements.txt
```


### ความต้องการของระบบ


- Python 3.10+
- แนะนำให้มี CUDA GPU แต่ไม่บังคับ (`Config.DEVICE` ตกไปใช้ CPU อัตโนมัติ)


ติดตั้ง dependency:


```bash
pip install -r requirements.txt
```


Package หลักที่ใช้: `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python`, `pillow`, `matplotlib`, 
`seaborn`, `tqdm`, `rich`


### โครงสร้าง Dataset


ชี้ `Config.DATA_ROOT` ไปที่โฟลเดอร์ที่มีสองโฟลเดอร์ย่อย (ตั้งชื่อได้ผ่าน `GOOD_DIRNAME` / `DEFECT_DIRNAME`):


```
<DATA_ROOT>/
├── good/      # ภาพปกติ (ใช้เทรน autoencoder)
└── defect/    # ภาพที่มีตำหนิ/ผิดปกติ
```


นามสกุลไฟล์ที่รองรับโดย default: `.jpg .jpeg .png .bmp` (`Config.VALID_EXT`) ภาพถูกแบ่ง 70/15/15 เป็น train/val/test โดย default 
(`Config.SPLIT_RATIOS`) การแบ่งจะถูก cache ไว้ที่ `Config.SPLIT_CACHE_PATH` เพื่อให้รันซ้ำแล้วได้ผลเหมือนเดิม **มีแต่ภาพ "good" เท่านั้นที่จะเข้า 
training set** — มีการ assert ตรวจสอบเรื่องนี้ตอนแบ่ง split


### วิธีใช้งาน


#### เริ่มต้นเร็ว


`RUN.py` override ค่า default ของ `Config` (path ของ dataset, hyperparameter ฯลฯ) โดยไม่ต้องแก้ `config/config.py` แล้วรัน train ตามด้วย 
visualize:


```bash
python RUN.py
```


แก้ dict `OVERRIDES` ที่ต้นไฟล์ `RUN.py` (หรือแก้ `config/config.py` ตรงๆ ก็ได้) เพื่อชี้ไปที่ dataset ของคุณและปรับ hyperparameter — ทุก 
hyperparameter ของ loss/model/training ถูกเปิดให้ปรับตรงนี้หมด ไม่ต้องแก้ไฟล์ใน `src/` เลย


#### รันทีละขั้นตอน


```bash
# 1. เทรน autoencoder, คำนวณ score val/test, เซฟ artifact ทั้งหมด
python scripts/train.py


# 2. render กราฟ, heatmap, และ gallery ภาพจาก artifact ที่เซฟไว้
python scripts/visualize.py
```


`scripts/visualize.py` แค่ *อ่าน* artifact ที่ `scripts/train.py` เซฟไว้ (history, scores, checkpoint, threshold, norm stats) เท่านั้น — 
ไม่เคยเทรนหรือคำนวณ score ใหม่เลย ถ้า artifact หายจะ raise error บอกให้รัน `train.py` ก่อน


### ตัวเลือก config ที่สำคัญ


ทุก option อยู่ใน `config/config.py` (dataclass `Config`) ดูรายละเอียดเต็มของแต่ละ field ได้ในไฟล์นั้น ที่สำคัญ:


| ตัวเลือก | ค่าที่เป็นไปได้ |
|---|---|
| `BACKBONE` | `tiny` \| `small` \| `base` \| `large` (ConvNeXt variant) |
| `LOSS` | `MSE` \| `MAE`/`L1` \| `HUBER`/`SMOOTH_L1` \| `COS` \| `COS_MSE` |
| `OPTIM` | `Adam` \| `AdamW` \| `SGD` \| `RMSprop` |
| `SCORE_METHOD` | `mean` \| `max` \| `topk` (วิธีรวม score ระดับภาพ) |
| โหมดสี | `RGB` (default) \| `GRAYSCALE` \| `GRAYSCALE_EQUALIZATION` \| `GRAYSCALE_CLAHE` \| `GRAYSCALE_EQUALIZATION_CLAHE` — 
ควบคุมผ่าน `USE_GRAYSCALE`, `USE_GRAYSCALE_EQUALIZATION`, `USE_CLAHE` |


#### Loss functions


Autoencoder เทรนบน feature map (ConvNeXt activation ที่ z-score normalize แล้วต่อ channel) ไม่ใช่ pixel ดิบๆ `COS`/`COS_MSE` ถูกเพิ่มเข้ามา
โดยเฉพาะ เพราะ loss แบบ pixel-space อย่าง SSIM มีสมมติฐาน (luminance term, `data_range` ที่ต้อง calibrate, window size ที่ปรับตามความ
ละเอียดภาพ) ที่ไม่ transfer มาที่ feature space ที่ normalize แล้วได้ตรงๆ — ดูเหตุผลเต็มได้ใน `loss_functions_summary.md` (SSIM/SSIM_MSE ถูกถอด
ออกจากโค้ดนี้ด้วยเหตุผลนี้) ตัวเลือกที่มี:


| ค่า `LOSS` | Config field ที่เกี่ยวข้อง | หมายเหตุ |
|---|---|---|
| `MSE` | — | ค่า default squared error ต่อ element แบบมาตรฐาน |
| `MAE` / `L1` | — | ทนต่อ outlier มากกว่า MSE, gradient คงที่ |
| `HUBER` / `SMOOTH_L1` | `HUBER_DELTA` | quadratic ตอนต่ำกว่า `HUBER_DELTA`, linear ตอนสูงกว่า — เหมาะสมเพราะ `HUBER_DELTA` มีความหมาย
เชิงกายภาพตรงๆ (จำนวน SD) บน feature ที่ z-scored แล้ว |
| `COS` | `COS_EPS` | cosine distance ต่อพิกเซลข้ามแกน channel — มองเวกเตอร์ยาว C ที่แต่ละตำแหน่งเป็นทิศทางหนึ่ง ไม่ต้องมีสมมติฐานเรื่อง 
`data_range`/luminance เลยจึง stationary ตลอดการเทรน scale-invariant โดยธรรมชาติ: เดี่ยวๆ แล้วไม่เคยลงโทษการเลื่อนของขนาด reconstruction 
เลย ลงโทษแค่ทิศทาง |
| `COS_MSE` | `COS_LAM`, `COS_EPS` | ผสม cosine distance (ทิศทาง / pattern การ activate ข้าม channel) กับ MSE (ขนาด) `COS_LAM=1.0` 
ลดรูปเป็น `COS` ล้วน, `COS_LAM=0.0` ลดรูปเป็น `MSE` ล้วน แนะนำเป็นค่า default สำหรับ reconstruction บน feature space — ดูการเปรียบเทียบเต็มใน 
`loss_functions_summary.md` |


`Config.__post_init__` validate `OPTIM`, `LOSS` (เทียบกับทุก alias ที่ `get_criterion()` รู้จัก), และ `SPLIT_RATIOS` (ต้องรวมกัน = 1.0 และ
มีค่าครบ 3 ค่า) พร้อม fail fast ด้วย error message ที่แก้ปัญหาได้จริงถ้า `DATA_ROOT` ยังเป็นค่า placeholder หรือไม่มีอยู่จริง


criterion ตัวเดียวกันถูกใช้ทั้งตอนคำนวณ gradient ระหว่างเทรน (`train_autoencoder`) และตอนคำนวณ error map ที่ป้อนเข้า heatmap/score ที่รายงานจริง
ของ val/test (`score_dataset_split` → `elementwise_error_map`) — ทั้งสองจุดเรียกผ่าน dispatch table เดียวใน 
`src/losses.py:elementwise_error_map()` เท่านั้น จึงไม่มีทางเบี่ยงเบนออกจากกันเงียบๆ ได้


### Output


Artifact ถูกเขียนไว้ใต้ `Config.SAVE_PATH` (checkpoint/log) และ `Config.OUTPUT_PATH` (ผลลัพธ์) ได้แก่:


- `extractor_norm_stats.pt` — ค่าสถิติ normalize ที่ fit แล้ว
- `history.json` — loss & AUROC ของ train/validation ต่อ epoch
- `scores_val.npz`, `scores_test.npz` — score, label, heatmap ต่อภาพ
- `threshold.json` — threshold สำหรับ deployment + oracle diagnostic
- `predictions_val.csv`, `predictions_test.csv` — prediction ต่อภาพ
- `final_results.json` — สรุปผล run แบบเต็ม (snapshot ของทุก field ใน Config ผ่าน `dataclasses.asdict()` บวก metrics) สำหรับเทียบ 
experiment
- Checkpoint ของ autoencoder ที่เทรนแล้ว (`.pth`)


Console จะ print ตาราง live ที่อัปเดตผลลัพธ์ (AUC-ROC, Average Precision, Accuracy, Precision, Recall, F1) ของ validation กับ test 
split ด้วย


---

