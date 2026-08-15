"""One-shot entry point: ตั้งค่า Config default ด้วย OVERRIDES ด้านล่าง (ไม่แก้
config/config.py) แล้วรัน train → visualize ต่อกันในคำสั่งเดียว

One-shot entry point: patches Config defaults via OVERRIDES below (without
touching config/config.py), then runs train → visualize back to back.
"""
import torch

import scripts.train as train
import scripts.visualize as visualize
from config.config import Config

OVERRIDES = dict(
    # ── Data & paths / ข้อมูลและ path ──────────────────────────────────────
    DATA_ROOT="/config/thesis/data/group1",
    GOOD_DIRNAME="all_good",
    DEFECT_DIRNAME="all_defect",

    SPLIT_RATIOS=(0.70, 0.15, 0.15),          # (train, val, test) ต้องรวมกัน = 1.0 / must sum to 1.0
    SPLIT_CACHE_PATH="splits/split_assignment.csv",
    GROUP_ID_REGEX=None,                      # regex 1 capture group กัน sample เดียวกันหลุดคนละ split / keeps same-group samples out of different splits
    SAVE_PATH="/config/thesis/result/PH-0/save",
    OUTPUT_PATH="/config/thesis/result/PH-0/output",
    VALID_EXT=('.jpg', '.jpeg', '.png', '.bmp'),

    # ── Reproducibility / ทำซ้ำผลได้ ────────────────────────────────────────
    SEED=42,
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    EXPERIMENT='PH-0',  # ชื่อ experiment ที่จะถูกเก็บใน final_results.json / experiment name saved into final_results.json

    # ── Loss & optimizer / ฟังก์ชัน loss และ optimizer ──────────────────────
    LOSS='MSE',                                   # MSE | MAE/L1 | HUBER/SMOOTH_L1 | COS | COS_MSE
    HUBER_DELTA=1.0,                            # ใช้เมื่อ LOSS='HUBER'/'SMOOTH_L1' / used when LOSS='HUBER'/'SMOOTH_L1'
    COS_LAM=0.5,                                 # ใช้เมื่อ LOSS='COS_MSE' (weight ของ cosine term, 1-COS_LAM = weight ของ MSE term) / used when LOSS='COS_MSE' (cosine-term weight; 1-COS_LAM = MSE-term weight)
    COS_EPS=1e-8,                                # ใช้เมื่อ LOSS='COS'/'COS_MSE' (stability epsilon ปกติไม่ต้องปรับ) / used when LOSS='COS'/'COS_MSE' (stability epsilon, rarely needs tuning)
    OPTIM='Adam',                               # Adam | AdamW | SGD | RMSprop
    AE_MOMENTUM=0.9,                            # ใช้เมื่อ OPTIM='SGD'/'RMSprop' / used when OPTIM='SGD'/'RMSprop'
    AE_SGD_NESTEROV=True,                       # ใช้เมื่อ OPTIM='SGD' / used when OPTIM='SGD'
    AE_RMSPROP_ALPHA=0.99,                      # ใช้เมื่อ OPTIM='RMSprop' / used when OPTIM='RMSprop'
    AE_RMSPROP_EPS=1e-8,                        # ใช้เมื่อ OPTIM='RMSprop' / used when OPTIM='RMSprop'

    # ── Backbone & model / โมเดลหลัก ────────────────────────────────────────
    BACKBONE='tiny',                            # tiny | small | base | large
    IMAGE_SIZE=(224, 224),

    # ── DataLoader ──────────────────────────────────────────────────────
    BATCH_SIZE=32,
    NUM_WORKERS=0,
    PIN_MEMORY=False,

    # ── Autoencoder training / เทรน autoencoder ─────────────────────────────
    AE_EPOCHS=100,
    AE_LR=1e-4,
    AE_WEIGHT_DECAY=5e-4,
    AE_BOTTLENECK_CH=64,
    AE_LR_STEP=25,                              # StepLR: ลด LR ทุก N epoch / StepLR: decay LR every N epochs
    AE_LR_GAMMA=0.5,                             # StepLR: ตัวคูณตอนลด LR / StepLR: decay multiplier
    AE_PATIENCE=20,                              # EarlyStopping patience (epoch) / EarlyStopping patience, in epochs

    # ── Heatmap & scoring / heatmap และการให้คะแนน ───────────────────────────
    HEATMAP_SIGMA=4.0,                          # Gaussian blur sigma ตอน upsample error map / Gaussian blur sigma when upsampling the error map
    THRESHOLD_PERCENTILE=95.0,                  # percentile ของ val-normal score ที่ใช้เป็น threshold / percentile of val-normal scores used as the threshold
    SCORE_METHOD='topk',                        # mean | max | topk
    SCORE_TOPK_PERCENT=10.0,                    # ใช้เมื่อ SCORE_METHOD='topk' / used when SCORE_METHOD='topk'
    AE_MONITOR='val_auroc',                     # val_auroc | val_loss_normal | val_loss
    USE_AUGMENTATION=True,
    AUG_COLOR_JITTER=0.20,

    # ── Preprocessing / color mode / การเตรียมภาพและโหมดสี ──────────────────
    # (ดูรายละเอียดลำดับความสำคัญของ 3 ตัวนี้ใน config/config.py::COLOR_MODE)
    # (see config/config.py::COLOR_MODE for the priority order of these 3 flags)
    USE_GRAYSCALE=False,
    USE_GRAYSCALE_EQUALIZATION=False,
    USE_CLAHE=False,
    CLAHE_CLIP_LIMIT=2.0,                       # ใช้เมื่อ USE_CLAHE=True / used when USE_CLAHE=True
    CLAHE_TILE_GRID_SIZE=(8, 8),                # ใช้เมื่อ USE_CLAHE=True / used when USE_CLAHE=True
)

_original_init = Config.__init__


def _patched_init(self, *args, **kwargs):
    """แทรกค่าจาก OVERRIDES เป็น default ให้ Config() ทุกครั้งที่ถูกเรียก
    โดยไม่แก้ config.py — ค่าที่ผู้เรียกใส่มาเองยังคงมีสิทธิ์เหนือกว่าเสมอ
    (kwargs.setdefault จะไม่ทับค่าที่ระบุมาแล้ว)

    Injects OVERRIDES as defaults into every Config() call, without
    touching config.py — values the caller passes explicitly always take
    priority (kwargs.setdefault never overwrites an already-given value).
    """
    for key, value in OVERRIDES.items():
        kwargs.setdefault(key, value)
    _original_init(self, *args, **kwargs)


Config.__init__ = _patched_init
# ── จบส่วนตั้งค่า / end of configuration section ────────────────────────────


if __name__ == "__main__":

    print("\n--- [1/2] เริ่มทำงาน Train ---")
    train.main()

    print("\n--- [2/2] เริ่มทำงาน Visualize ---")
    visualize.main()

    print("\n✅ เสร็จสิ้นกระบวนการทั้งหมดเรียบร้อย!")