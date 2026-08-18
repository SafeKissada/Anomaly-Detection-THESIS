# Anomaly Detection THESIS

🇹🇭 [ภาษาไทย](#ภาษาไทย) | 🇬🇧 [English](#english)

---

## ภาษาไทย

ตรวจจับความผิดปกติ/ตำหนิเชิงภาพ (visual anomaly / defect detection) ด้วย **ตัวสกัด feature ConvNeXt แบบ frozen** ร่วมกับ **convolutional autoencoder ที่เทรนได้** บน feature map ที่สกัดออกมา autoencoder เรียนรู้ที่จะ reconstruct feature ของภาพ *ปกติ (good)* เท่านั้น เมื่อถึงตอน inference reconstruction error จะถูกแปลงเป็น heatmap ระดับพิกเซลและ anomaly score ระดับภาพ ซึ่งจะถูกนำไป threshold เพื่อตัดสินว่าเป็นตำหนิหรือไม่

### วิธีการทำงาน

1. **สกัด feature** — ConvNeXt (`tiny` / `small` / `base` / `large`) ที่ pretrain มาแล้วและ frozen สกัด feature หลาย stage (Stage2 + Stage3 นำมาต่อกัน) จากแต่ละภาพ ค่าสถิติ mean/std ต่อ channel สำหรับ normalize ถูก fit เพียงครั้งเดียวจาก training set ที่เป็นภาพ *ปกติ* เท่านั้น
2. **Autoencoder** — encoder-decoder แบบ convolutional ที่เบา (ConvNeXt-style block พร้อม depthwise conv และ residual connection) บีบอัด feature ที่ normalize แล้วลงไปที่ bottleneck แล้ว reconstruct กลับ เทรน**เฉพาะภาพปกติ (good)** เท่านั้น
3. **คำนวณ anomaly score** — ตอน inference reconstruction error ต่อพิกเซล (`MSE` / `MAE` / `HUBER` / `COS` / `COS_MSE`, ปรับได้) จะถูก upsample, Gaussian-smooth เป็น heatmap แล้วรวมเป็น score เดียวต่อภาพ (`mean` / `max` / `top-k%`)
4. **ตั้ง threshold** — threshold สำหรับ deployment เลือกจาก percentile ของ score ภาพปกติใน validation set ส่วน oracle threshold (max-F1) คำนวณไว้เป็น diagnostic เท่านั้น — **ไม่ได้ใช้** ในตัวเลขที่รายงานจริง เพราะใช้ label ของ validation set ซึ่งไม่ควรรู้ตอน deploy จริง
5. **ประเมินผลและ visualize** — คำนวณ AUC-ROC, average precision, accuracy, precision, recall, F1 บน val/test แยกออกมา render กราฟ ROC/PR, confusion matrix, score distribution, training curve, และ gallery ภาพ/heatmap ต่างหาก

### Anomaly Detection Paradigm

ระบบนี้ใช้แนวทาง **semi-supervised (one-class) anomaly detection** — ไม่ใช่ unsupervised แท้ และไม่ใช่ supervised แท้ แต่ละ phase ใช้ label ต่างกัน:

| Phase | ใช้ label ไหม | ลักษณะ |
|---|---|---|
| **Training** (`train_autoencoder()` เห็นแค่ `normal_loader`) | **ไม่ใช้เลย** — AE ไม่เคยเห็นภาพ anomaly แม้แต่ภาพเดียวตอนเทรน | Unsupervised reconstruction learning บน training set ที่ถูกกรองไว้แล้วว่าเป็นของปกติล้วนๆ |
| **Validation — เลือก checkpoint** | **ขึ้นกับ `AE_MONITOR`** — `val_loss` (ค่าที่ตั้งใน `RUN.py`) ไม่ใช้ label เลย ส่วน `val_auroc`/`val_loss_normal` ใช้ label ในระดับต่างกัน (ดู [Validation Monitoring](#validation-monitoring-ae_monitor)) | label-free ได้ ถ้าเลือก `val_loss` |
| **Validation — ตั้ง threshold** | **ใช้เสมอ** — `select_percentile_threshold()` ต้องกรองเอาเฉพาะ score ภาพ normal มาคำนวณ percentile ไม่ขึ้นกับ `AE_MONITOR` | ต้องมี label ของ validation set เสมอ |
| **Test / Inference** (deploy จริง) | **ไม่ใช้** — โมเดลทำนายจาก reconstruction error อย่างเดียว label ใช้แค่วัดผลย้อนหลัง | Unsupervised ที่ inference time |

ตามการแบ่งของวรรณกรรม anomaly detection (เช่น Ruff et al. 2021, *"A Unifying Review of Deep and Shallow Anomaly Detection"*) มี 3 แบบหลัก:

- **Unsupervised AD** — training set เป็นข้อมูลผสมที่ไม่รู้ label เลย (normal ปนกับ anomaly โดยไม่รู้ว่าอันไหนเป็นอะไร)
- **Semi-supervised AD (one-class)** — training set มีแต่ normal เท่านั้น (รู้แน่ชัดว่าสะอาด) — **นี่คือสิ่งที่ระบบนี้ทำ**: `scan_and_split()` การันตีว่า `train` split มีแต่แถว `label == 'normal'` เท่านั้น พร้อม `_assert_no_defect_in_train()` เป็นตาข่ายนิรภัยตรวจซ้ำ
- **Supervised AD** — เทรนด้วยทั้ง normal และ anomaly ที่มี label ครบ เป็น binary classification ตรงๆ (ระบบนี้**ไม่ใช่**แบบนี้)

**จุดที่ต้อง honest**: ไม่ claim ว่าทั้ง pipeline เป็น "unsupervised 100%" แม้จะตั้ง `AE_MONITOR='val_loss'` (ค่า default ของ `RUN.py`) ให้การเลือก checkpoint ระหว่างเทรนเป็น label-free แล้วก็ตาม เพราะยังมีอีก 2 จุดที่ใช้ label เสมอไม่ว่า `AE_MONITOR` จะตั้งเป็นอะไร: (1) `scan_and_split()` ต้องรู้ label good/defect ตอนคัด train/val/test ตั้งแต่ต้น และ (2) `select_percentile_threshold()`/`compute_metrics()` ต้องใช้ label ของ validation/test set เพื่อตั้ง threshold และรายงานผล AUROC/F1 — ระบบนี้จึงยังคงเป็น **semi-supervised (one-class)** เสมอ ไม่ว่าจะตั้ง `AE_MONITOR` เป็นอะไร ส่วน `oracle_threshold_diagnostic()` (max-F1) ใช้ label ของ anomaly ใน validation ตรงๆ ด้วย — แต่เป็น **diagnostic เท่านั้น ไม่ได้ใช้เป็นผลจริง** เพราะจะทำให้ deployment ไม่ realistic

สรุป: *"ระบบนี้ใช้แนวทาง semi-supervised (one-class) anomaly detection — autoencoder ถูกเทรนด้วยภาพปกติ (normal) เท่านั้น โดยไม่เคยเห็นภาพที่มีตำหนิ (anomaly) ระหว่างการเทรนเลย ด้วยการตั้ง `AE_MONITOR='val_loss'` การเลือก checkpoint ระหว่างเทรนก็เป็น label-free เช่นกัน label ของภาพ anomaly ถูกใช้เฉพาะขั้นตอนตั้ง deployment threshold และวัดผลบน validation/test set เท่านั้น ไม่ได้ถูกใช้ในการคำนวณ loss function หรือปรับ parameter ของโมเดลแต่อย่างใด"*

### โครงสร้างโปรเจกต์

```
.
├── RUN.py                     # entry point เดียวจบ: override ค่า Config แล้วรัน train → visualize
├── config/
│   └── config.py               # dataclass Config รวมทุก hyperparameter + validation ใน __post_init__
├── scripts/
│   ├── train.py                 # เทรน autoencoder → คำนวณ score val/test → เซฟ artifact ทั้งหมด
│   ├── visualize.py             # โหลด artifact ที่เซฟไว้ → render กราฟ/heatmap/gallery ทั้งหมด
│   └── run_cost_aware.py         # cost-aware threshold sweep — รันแยกจาก train.py ใช้ artifact เดิม
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
│   ├── cost_aware.py             # cost-aware threshold selection (r·FN + FP) — ทางเลือกแทน percentile/F1 threshold
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

Package หลักที่ใช้: `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, `rich`

### โครงสร้าง Dataset

ชี้ `Config.DATA_ROOT` ไปที่โฟลเดอร์ที่มีสองโฟลเดอร์ย่อย (ตั้งชื่อได้ผ่าน `GOOD_DIRNAME` / `DEFECT_DIRNAME`):

```
<DATA_ROOT>/
├── all_good/      # ภาพปกติ (ใช้เทรน autoencoder)
└── all_defect/    # ภาพที่มีตำหนิ/ผิดปกติ
```

นามสกุลไฟล์ที่รองรับโดย default: `.jpg .jpeg .png .bmp` (`Config.VALID_EXT`) ภาพถูกแบ่ง 70/15/15 เป็น train/val/test โดย default (`Config.SPLIT_RATIOS`) การแบ่งจะถูก cache ไว้ที่ `Config.SPLIT_CACHE_PATH` เพื่อให้รันซ้ำแล้วได้ผลเหมือนเดิม **มีแต่ภาพ "good" เท่านั้นที่จะเข้า training set** — มีการ assert ตรวจสอบเรื่องนี้ตอนแบ่ง split

### วิธีใช้งาน

#### เริ่มต้นเร็ว

`RUN.py` override ค่า default ของ `Config` (path ของ dataset, hyperparameter ฯลฯ) โดยไม่ต้องแก้ `config/config.py` แล้วรัน train ตามด้วย visualize:

```bash
python RUN.py
```

แก้ dict `OVERRIDES` ที่ต้นไฟล์ `RUN.py` (หรือแก้ `config/config.py` ตรงๆ ก็ได้) เพื่อชี้ไปที่ dataset ของคุณและปรับ hyperparameter — ทุก hyperparameter ของ loss/model/training ถูกเปิดให้ปรับตรงนี้หมด ไม่ต้องแก้ไฟล์ใน `src/` เลย

#### รันทีละขั้นตอน

```bash
# 1. เทรน autoencoder, คำนวณ score val/test, เซฟ artifact ทั้งหมด
python scripts/train.py

# 2. render กราฟ, heatmap, และ gallery ภาพจาก artifact ที่เซฟไว้
python scripts/visualize.py
```

`scripts/visualize.py` แค่ *อ่าน* artifact ที่ `scripts/train.py` เซฟไว้ (history, scores, checkpoint, threshold, norm stats) เท่านั้น — ไม่เคยเทรนหรือคำนวณ score ใหม่เลย ถ้า artifact หายจะ raise error บอกให้รัน `train.py` ก่อน

### ตัวเลือก config ที่สำคัญ

ทุก option อยู่ใน `config/config.py` (dataclass `Config`) ดูรายละเอียดเต็มของแต่ละ field ได้ในไฟล์นั้น ที่สำคัญ:

| ตัวเลือก | ค่าที่เป็นไปได้ |
|---|---|
| `BACKBONE` | `tiny` \| `small` \| `base` \| `large` (ConvNeXt variant) |
| `LOSS` | `MSE` \| `MAE`/`L1` \| `HUBER`/`SMOOTH_L1` \| `COS` \| `COS_MSE` |
| `OPTIM` | `Adam` \| `AdamW` \| `SGD` \| `RMSprop` |
| `SCORE_METHOD` | `mean` \| `max` \| `topk` |`structure` (วิธีรวม score ระดับภาพ) |
| โหมดสี | `RGB` (default) \| `GRAYSCALE` \| `GRAYSCALE_EQUALIZATION` \| `GRAYSCALE_CLAHE` \| `GRAYSCALE_EQUALIZATION_CLAHE` — ควบคุมผ่าน `USE_GRAYSCALE`, `USE_GRAYSCALE_EQUALIZATION`, `USE_CLAHE` |

#### Loss functions

Autoencoder เทรนบน feature map (ConvNeXt activation ที่ z-score normalize แล้วต่อ channel) ไม่ใช่ pixel ดิบๆ `COS`/`COS_MSE` ถูกเพิ่มเข้ามาโดยเฉพาะ เพราะ loss แบบ pixel-space อย่าง SSIM มีสมมติฐาน (luminance term, `data_range` ที่ต้อง calibrate, window size ที่ปรับตามความละเอียดภาพ) ที่ไม่ transfer มาที่ feature space ที่ normalize แล้วได้ตรงๆ — ดูเหตุผลเต็มได้ใน `loss_functions_summary.md` (SSIM/SSIM_MSE ถูกถอดออกจากโค้ดนี้ด้วยเหตุผลนี้) ตัวเลือกที่มี:

| ค่า `LOSS` | Config field ที่เกี่ยวข้อง | หมายเหตุ |
|---|---|---|
| `MSE` | — | ค่า default squared error ต่อ element แบบมาตรฐาน |
| `MAE` / `L1` | — | ทนต่อ outlier มากกว่า MSE, gradient คงที่ |
| `HUBER` / `SMOOTH_L1` | `HUBER_DELTA` | quadratic ตอนต่ำกว่า `HUBER_DELTA`, linear ตอนสูงกว่า — เหมาะสมเพราะ `HUBER_DELTA` มีความหมายเชิงกายภาพตรงๆ (จำนวน SD) บน feature ที่ z-scored แล้ว |
| `COS` | `COS_EPS` | cosine distance ต่อพิกเซลข้ามแกน channel — มองเวกเตอร์ยาว C ที่แต่ละตำแหน่งเป็นทิศทางหนึ่ง ไม่ต้องมีสมมติฐานเรื่อง `data_range`/luminance เลยจึง stationary ตลอดการเทรน scale-invariant โดยธรรมชาติ: เดี่ยวๆ แล้วไม่เคยลงโทษการเลื่อนของขนาด reconstruction เลย ลงโทษแค่ทิศทาง |
| `COS_MSE` | `COS_LAM`, `COS_EPS` | ผสม cosine distance (ทิศทาง / pattern การ activate ข้าม channel) กับ MSE (ขนาด) `COS_LAM=1.0` ลดรูปเป็น `COS` ล้วน, `COS_LAM=0.0` ลดรูปเป็น `MSE` ล้วน แนะนำเป็นค่า default สำหรับ reconstruction บน feature space — ดูการเปรียบเทียบเต็มใน `loss_functions_summary.md` |

`Config.__post_init__` validate `OPTIM`, `LOSS` (เทียบกับทุก alias ที่ `get_criterion()` รู้จัก), และ `SPLIT_RATIOS` (ต้องรวมกัน = 1.0 และมีค่าครบ 3 ค่า) พร้อม fail fast ด้วย error message ที่แก้ปัญหาได้จริงถ้า `DATA_ROOT` ยังเป็นค่า placeholder หรือไม่มีอยู่จริง

criterion ตัวเดียวกันถูกใช้ทั้งตอนคำนวณ gradient ระหว่างเทรน (`train_autoencoder`) และตอนคำนวณ error map ที่ป้อนเข้า heatmap/score ที่รายงานจริงของ val/test (`score_dataset_split` → `elementwise_error_map`) — ทั้งสองจุดเรียกผ่าน dispatch table เดียวใน `src/losses.py:elementwise_error_map()` เท่านั้น จึงไม่มีทางเบี่ยงเบนออกจากกันเงียบๆ ได้

#### Validation Monitoring (`AE_MONITOR`)

หลังจบทุก epoch ตอนเทรน (`train_autoencoder()` ใน `src/engine.py`) จะคำนวณ validation metric ออกมา 3 ตัวพร้อมกัน ไม่ใช่แค่ตัวเดียว — ทั้ง 3 ตัวใช้ label ต่างระดับกัน ตรวจจากโค้ดจริง (`src/engine.py`) ดังนี้:

| Key ใน `history` | คำนวณยังไง | ใช้ label ไหม | วัดอะไร |
|---|---|---|---|
| `val_loss` | `criterion(recon, feats)` บน validation set **ทั้งชุด** (ผสม normal + anomaly) โดยไม่แตะ `batch_labels` เลย | **ไม่ใช้เลย** — label-free / unsupervised | reconstruction error โดยรวม ไม่แยกว่าภาพไหนเป็นของเสีย |
| `val_loss_normal` | อ่าน `batch_labels` เพื่อสร้าง `normal_mask` ก่อน แล้วเฉลี่ย loss เฉพาะแถวที่ label='normal' | **ใช้แบบเบา** — รู้แค่ว่าภาพไหนเป็น known-normal (label เดียวกับที่ใช้คัด training set ตั้งแต่ต้น ไม่ใช่ anomaly label) | สุขภาพของ autoencoder ล้วนๆ ไม่ปนกับสัญญาณจาก anomaly |
| `val_auroc` | error map ผ่าน pipeline เดียวกับ scoring จริง (`upsample_and_smooth` + `aggregate_score`) แล้วคำนวณ AUROC เทียบ label จริง (ต้องมีทั้ง 2 class) | **ใช้เต็มรูป** — ต้องมี anomaly label ครบ | ความสามารถในการ**แยกแยะ**ของ/ปกติ — ตรงกับสิ่งที่วัดผลจริงที่สุด |

**`val_loss` คือตัวเดียวที่เป็น unsupervised แท้ตามนิยามเข้มงวด** (ไม่อ่าน label เลยแม้แต่บรรทัดเดียวตอนคำนวณ) — ถ้าตั้งใจออกแบบให้ validation loop ทั้งหมดไม่แตะ label เลย ต้องเลือก `AE_MONITOR='val_loss'` เท่านั้น `val_loss_normal` แม้จะไม่ใช้ anomaly label แต่ก็ยังต้องรู้ว่าแถวไหนคือ "known-normal" ถึงจะกรองมาเฉลี่ยได้ จึงไม่ใช่ label-free 100% ในทางเทคนิค (แม้ label ที่ใช้จะเป็นประเภทเดียวกับที่ใช้คัด training set มาตั้งแต่ต้น ไม่ใช่ anomaly-specific information แบบที่ `val_auroc` ต้องใช้)

`cfg.AE_MONITOR` (`'val_auroc'` | `'val_loss_normal'` | `'val_loss'`) เลือกว่าจะใช้ตัวไหนตัดสินใจ ค่านี้ถูกใช้ 2 จุดที่ต้องสอดคล้องกันเสมอ:

1. **`EarlyStopping`** — `mode` ถูก map อัตโนมัติตาม monitor (`val_auroc` ใช้ `mode='max'` เพราะยิ่งสูงยิ่งดี ส่วนอีกสองตัวใช้ `mode='min'` เพราะยิ่งต่ำยิ่งดี) ไม่ต้องตั้งเองแยก
2. **`get_best_epoch()`** ใน `src/engine.py` — ฟังก์ชันกลางที่ทั้ง `scripts/train.py` (รายงาน best epoch ใน `final_results.json`) และ `src/visual.py` (วาดเส้น best epoch บนกราฟ) เรียกใช้ร่วมกัน รับประกันว่าสอง จุดนี้ได้คำตอบตรงกันเป๊ะ

**`Config` dataclass default vs. ค่าที่ใช้จริงใน `RUN.py`**: `config/config.py` ตั้ง `AE_MONITOR = 'val_auroc'` เป็น dataclass default (เลือกผลลัพธ์การแยกแยะที่ดีที่สุด) แต่ `RUN.py` (entry point หลักที่แนะนำให้ใช้) override เป็น `AE_MONITOR = 'val_loss'` แทน เพื่อให้การเลือก checkpoint ระหว่างเทรนเป็น **unsupervised อย่างเคร่งครัด** — `val_loss` ต่ำ ไม่ได้แปลว่าโมเดลดีเสมอไป (AE ที่ reconstruct แม่นมากจนรวมถึง anomaly ด้วยจะทำให้ error ของ normal/anomaly แยกกันไม่ออก ซึ่งกลับทำให้ AUROC แย่ลง) แต่เป็น trade-off ที่ตั้งใจเลือก: **unsupervised อย่างเคร่งครัด** (`val_loss`) แลกกับ **ผลลัพธ์การแยกแยะที่ดีที่สุด** (`val_auroc`) — ปรับกลับไปใช้ `val_auroc` ได้จาก `RUN.py` ถ้าต้องการเปรียบเทียบทั้งสองแบบเป็น ablation

### Cost-Aware Threshold Selection (`src/cost_aware.py`)

**สถานะ**: candidate novelty มุมที่ 2 (อีกตัวคือ Leave-One-Group-Out) — ยังไม่ได้ล็อกว่าเป็น novelty หลัก และยังไม่เคยรันกับข้อมูลจริง

Threshold แบบ percentile เดิม (`select_percentile_threshold`) ไม่สนใจว่า defect จริงจะหลุดไปกี่ชิ้น — AUC สูงไม่ได้แปลว่า threshold ที่เลือกใช้จริงจะดี เพราะ AUC วัดคุณภาพ ranking ของ score ทั้งหมด ไม่ขึ้นกับ threshold ตัวใดตัวหนึ่ง F1-based threshold ก็มีปัญหาเดียวกันเชิงโครงสร้าง เพราะถ่วงน้ำหนัก FP/FN เท่ากันเป๊ะ ทั้งที่ในบริบท QC จริง **FN (escape) แพงกว่า FP (false alarm) มาก**

`src/cost_aware.py` เลือก threshold จาก total cost แทน:

```
Total Cost(t, r) = r · FN(t) + 1 · FP(t)
t*(r) = argmin_t [ r·FN(t) + FP(t) ]
```

`r` คือ cost ratio (escape 1 ชิ้นแพงเท่ากับ false-check กี่ครั้ง) ใช้ parametric ratio แทนตัวเลขต้นทุนจริง (บาท/นาที) เพราะไม่มีข้อมูลต้นทุนจริงให้ใช้ — generalize ข้ามบริบท deployment ได้ ไม่ผูกกับ cost structure ของโรงงานเดียว

รันแยกจาก `train.py` โดยสิ้นเชิง ใช้ artifact ที่มีอยู่แล้ว (`scores_val.npz`, `scores_test.npz`) ไม่ต้อง train ใหม่:

```bash
python scripts/train.py          # ทำครั้งเดียว ได้ artifact ครบ
python scripts/run_cost_aware.py # รันกี่ครั้งก็ได้ ไม่ต้อง train ซ้ำ
```

**ไม่มี `r` ที่ "ถูกต้อง" ตายตัว** — framework นี้ให้เครื่องมือเลือก ไม่ใช่ตัวเลขสำเร็จรูป มี 3 แนวทาง:

| แนวทาง | ฟังก์ชัน | เมื่อไหร่ควรใช้ |
|---|---|---|
| แสดงทั้ง curve ไม่เลือก | `cost_sweep_report()` | Framework paper — ให้ผู้อ่าน/โรงงานเลือกเองจาก cost ratio จริง |
| Elbow point (ตัวอย่างประกอบเท่านั้น) | `find_elbow_r()` | ต้องมีตัวเลขตัวแทนสักตัวในเล่ม — ต้องระบุชัดว่าเป็นตัวอย่าง ไม่ใช่ deployment number |
| Recall-constrained | `select_recall_constrained_threshold()` | มี business spec อยู่แล้ว (เช่น "escape ต้อง ≤5%") ไม่ต้องรู้ cost เป็นเงินเลย |

**Known limitations**:
- Candidate ของ `sweep_thresholds()`/`select_recall_constrained_threshold()` ไม่รวม threshold ที่ "ไม่ flag เลย" (เหมือน `oracle_threshold_diagnostic()` เดิมที่ตัด endpoint สุดท้ายทิ้ง) — ในทางปฏิบัติแทบไม่กระทบเพราะ `r` ที่ใช้จริง (1–100) มักให้ threshold ต่ำอยู่แล้ว
- ที่ `r=1`, minimize `FN+FP` เทียบเท่า maximize accuracy **ไม่ใช่** maximize F1 — ตรงกันแค่ตอน class balance เท่านั้น **ห้ามใช้ "r=1 ต้องใกล้ max-F1 threshold" เป็น correctness check** (พิสูจน์แล้วด้วย simulation ว่าต่างกันได้โดยไม่มีบั๊ก) ใช้ monotonicity check แทน: `r` เพิ่ม → threshold ไม่เพิ่ม → escape_rate ไม่เพิ่ม

### Output

Artifact ถูกเขียนไว้ใต้ `Config.SAVE_PATH` (checkpoint/log) และ `Config.OUTPUT_PATH` (ผลลัพธ์) ได้แก่:

- `extractor_norm_stats.pt` — ค่าสถิติ normalize ที่ fit แล้ว
- `history.json` — loss & AUROC ของ train/validation ต่อ epoch
- `scores_val.npz`, `scores_test.npz` — score, label, heatmap ต่อภาพ
- `threshold.json` — threshold สำหรับ deployment + oracle diagnostic
- `predictions_val.csv`, `predictions_test.csv` — prediction ต่อภาพ
- `final_results.json` — สรุปผล run แบบเต็ม (snapshot ของทุก field ใน Config ผ่าน `dataclasses.asdict()` บวก metrics) สำหรับเทียบ experiment
- Checkpoint ของ autoencoder ที่เทรนแล้ว (`.pth`)

Console จะ print ตาราง live ที่อัปเดตผลลัพธ์ (AUC-ROC, Average Precision, Accuracy, Precision, Recall, F1) ของ validation กับ test split ด้วย

---

## English

Visual anomaly / defect detection using a **frozen ConvNeXt feature extractor** + a **trainable convolutional autoencoder** on the extracted feature maps. The autoencoder learns to reconstruct features of *normal (good)* images only; at inference time, reconstruction error is turned into a pixel-level heatmap and an image-level anomaly score, which is thresholded to flag defects.

### How it works

1. **Feature extraction** — A pretrained, frozen `torchvision` ConvNeXt (`tiny` / `small` / `base` / `large`) extracts multi-stage features (Stage2 + Stage3, concatenated) from each image. Per-channel mean/std normalization statistics are fitted once on the *normal* training set only.
2. **Autoencoder** — A lightweight convolutional encoder–decoder (ConvNeXt-style blocks with depthwise conv + residual connections) compresses the normalized features to a bottleneck and reconstructs them. It is trained **only on normal (good) images**.
3. **Anomaly scoring** — At inference, the per-pixel reconstruction error (`MSE` / `MAE` / `HUBER` / `COS` / `COS_MSE`, configurable) is upsampled, Gaussian-smoothed into a heatmap, and aggregated into a single image-level score (`mean` / `max` / `top-k%`).
4. **Thresholding** — A deployment threshold is chosen as a percentile of the validation normal-image scores. An oracle (max-F1) threshold is also computed as a diagnostic only — it is **not** used for reported metrics, since it uses validation anomaly labels.
5. **Evaluation & visualization** — AUC-ROC, average precision, accuracy, precision, recall, F1 are computed on val/test; ROC/PR curves, confusion matrices, score distributions, training curves, and image/heatmap galleries are rendered separately.

### Anomaly Detection Paradigm

This system follows a **semi-supervised (one-class) anomaly detection** approach — neither purely unsupervised nor purely supervised. Each phase uses labels differently:

| Phase | Uses labels? | Nature |
|---|---|---|
| **Training** (`train_autoencoder()` only ever sees `normal_loader`) | **Never** — the AE never sees a single anomalous image during training | Unsupervised reconstruction learning on a training set that's been pre-filtered to contain only normal data |
| **Validation — checkpoint selection** | **Depends on `AE_MONITOR`** — `val_loss` (the value set in `RUN.py`) uses no labels at all; `val_auroc`/`val_loss_normal` use labels to different degrees (see [Validation Monitoring](#validation-monitoring-ae_monitor)) | Can be label-free, if `val_loss` is selected |
| **Validation — threshold selection** | **Always** — `select_percentile_threshold()` must filter to normal-only scores to compute the percentile, regardless of `AE_MONITOR` | Always requires labeled validation data |
| **Test / Inference** (real deployment) | **No** — the model predicts from reconstruction error alone; labels are only used to score it after the fact | Unsupervised at inference time |

Per the standard taxonomy in the anomaly-detection literature (e.g. Ruff et al. 2021, *"A Unifying Review of Deep and Shallow Anomaly Detection"*), there are three main paradigms:

- **Unsupervised AD** — the training set is a mixed, unlabeled pool (normal and anomalous data mixed together with no label telling you which is which).
- **Semi-supervised AD (one-class)** — the training set contains normal data only, and is known to be clean. **This is what this system does**: `scan_and_split()` guarantees the `train` split contains only `label == 'normal'` rows, with `_assert_no_defect_in_train()` as a hard safety-net double-check.
- **Supervised AD** — trained on both normal and anomalous data with full labels, as a straightforward binary classification problem. This system is **not** this.

**A point worth being precise about in a thesis**: don't describe the whole pipeline as "100% unsupervised," even with `AE_MONITOR='val_loss'` (the `RUN.py` default) making checkpoint selection during training label-free. Two other points always use labels regardless of `AE_MONITOR`: (1) `scan_and_split()` needs the good/defect labels to build train/val/test in the first place, and (2) `select_percentile_threshold()`/`compute_metrics()` need validation/test labels to set the threshold and report AUROC/F1. This system therefore always remains **semi-supervised (one-class)**, whatever `AE_MONITOR` is set to. `oracle_threshold_diagnostic()` (max-F1) also uses validation anomaly labels directly — but it is explicitly a **diagnostic only, never used for reported results**, since relying on it would make the deployment scenario unrealistic.

A concise summary usable directly in a thesis: *"This system uses a semi-supervised (one-class) anomaly detection approach — the autoencoder is trained exclusively on normal images and never observes a defective image during training. With `AE_MONITOR='val_loss'`, checkpoint selection during training is also label-free. Anomaly labels are used only for setting the deployment threshold and for evaluation on the validation/test sets; they are never used to compute the loss function or update any model parameter."*

### Project structure

```
.
├── RUN.py                     # One-shot entry point: patches Config defaults, then runs train → visualize
├── config/
│   └── config.py               # Dataclass Config with every hyperparameter + validation in __post_init__
├── scripts/
│   ├── train.py                 # Train autoencoder → score val/test → save all artifacts
│   ├── visualize.py             # Load saved artifacts → render all plots/heatmaps/galleries
│   └── run_cost_aware.py         # Cost-aware threshold sweep — runs separately from train.py, reuses artifacts
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
│   ├── cost_aware.py             # Cost-aware threshold selection (r·FN + FP) — an alternative to the percentile/F1 threshold
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
├── all_good/      # normal images (used to train the autoencoder)
└── all_defect/    # defective / anomalous images
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

#### Validation Monitoring (`AE_MONITOR`)

After every training epoch, `train_autoencoder()` (in `src/engine.py`) computes three validation metrics at once, not just one — each uses labels to a different degree, confirmed against the actual code (`src/engine.py`):

| `history` key | How it's computed | Uses labels? | What it measures |
|---|---|---|---|
| `val_loss` | `criterion(recon, feats)` over the **entire** validation set (mixed normal + anomaly), never touching `batch_labels` | **Never** — label-free / unsupervised | Overall reconstruction error — doesn't distinguish which images are defective |
| `val_loss_normal` | Reads `batch_labels` to build a `normal_mask` first, then averages loss only over rows where label='normal' | **Lightly** — only knows which images are known-normal (the same label already used to curate the training set, not anomaly-specific information) | Pure autoencoder health, uncontaminated by the anomaly signal |
| `val_auroc` | Each image's error map is run through the exact same pipeline used for real scoring (`upsample_and_smooth` + `aggregate_score`), then AUROC is computed against the true labels (needs both classes) | **Fully** — requires complete anomaly labels | How well the model **separates** normal from anomalous — the closest proxy to what's actually being evaluated |

**`val_loss` is the only strictly unsupervised one** (it never reads a label during computation) — if the design intent is a validation loop that never touches labels at all, `AE_MONITOR='val_loss'` is the only choice that satisfies that. `val_loss_normal`, while never using an anomaly label, still needs to know which rows are "known-normal" to filter on, so it isn't 100% label-free in the strict technical sense (even though the label it uses is the same kind already used to curate the training set, not the anomaly-specific information `val_auroc` requires).

`cfg.AE_MONITOR` (`'val_auroc'` | `'val_loss_normal'` | `'val_loss'`) selects which one drives model selection. It's used at two call sites that must always agree:

1. **`EarlyStopping`** — its `mode` is mapped automatically from the monitor (`val_auroc` uses `mode='max'` since higher is better; the other two use `mode='min'` since lower is better). No separate setting needed.
2. **`get_best_epoch()`** in `src/engine.py` — a single shared function called by both `scripts/train.py` (to report the best epoch in `final_results.json`) and `src/visual.py` (to mark the best epoch on the training curve), guaranteeing the two agree exactly.

**`Config` dataclass default vs. the value actually used by `RUN.py`**: `config/config.py` sets `AE_MONITOR = 'val_auroc'` as the dataclass default (best separability). `RUN.py` (the recommended entry point) overrides this to `AE_MONITOR = 'val_loss'`, so checkpoint selection during training is **strictly unsupervised**. A low `val_loss` doesn't necessarily mean a good model — an autoencoder that reconstructs everything accurately, anomalies included, drives its loss down while making normal and anomalous errors indistinguishable, which actually hurts AUROC. This is a deliberate trade-off: **strict unsupervised selection** (`val_loss`) traded for **best detection performance** (`val_auroc`) — switch back to `val_auroc` in `RUN.py` if you want to compare both as an ablation.

### Cost-Aware Threshold Selection (`src/cost_aware.py`)

**Status**: novelty candidate #2 (the other is Leave-One-Group-Out) — not yet locked as the main novelty, and not yet run against real data.

The existing percentile threshold (`select_percentile_threshold`) ignores how many real defects would slip through — high AUC doesn't mean the threshold actually used in deployment is good, since AUC measures overall ranking quality, not any single threshold. F1-based thresholds have the same structural problem, weighting FP and FN exactly equally, even though in a real QC context **FN (escape) is far more costly than FP (false alarm)**.

`src/cost_aware.py` selects the threshold by total cost instead:

```
Total Cost(t, r) = r · FN(t) + 1 · FP(t)
t*(r) = argmin_t [ r·FN(t) + FP(t) ]
```

`r` is the cost ratio (one escape costs as much as how many false checks). A parametric ratio is used instead of real monetary costs, since no real cost data is available — this generalizes across deployment contexts instead of being tied to one factory's cost structure.

Runs completely separately from `train.py`, reusing existing artifacts (`scores_val.npz`, `scores_test.npz`) — no re-training needed:

```bash
python scripts/train.py          # once, produces all artifacts
python scripts/run_cost_aware.py # re-runnable any time, no re-training
```

**There is no single "correct" `r`** — this framework provides a tool for choosing, not a ready-made number. Three approaches:

| Approach | Function | When to use |
|---|---|---|
| Show the whole curve, pick nothing | `cost_sweep_report()` | Framework paper — let the reader/factory choose from their own real cost ratio |
| Elbow point (illustrative only) | `find_elbow_r()` | A single representative number is needed in the thesis — must be clearly labeled as an example, not a deployment number |
| Recall-constrained | `select_recall_constrained_threshold()` | A business spec already exists (e.g. "escape must stay ≤5%") — no need to know cost in monetary terms at all |

**Known limitations**:
- Candidates in `sweep_thresholds()`/`select_recall_constrained_threshold()` never include the "flag nothing" threshold (same as the existing `oracle_threshold_diagnostic()`, which drops the last endpoint) — rarely matters in practice since the `r` values actually used (1–100) tend to push the threshold low anyway.
- At `r=1`, minimizing `FN+FP` is equivalent to maximizing accuracy, **not** F1 — the two only coincide under class balance. **Never use "r=1 should match the max-F1 threshold" as a correctness check** (proven via simulation that they can legitimately differ with zero bugs). Use the monotonicity check instead: `r` increases → threshold non-increasing → escape_rate non-increasing.

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