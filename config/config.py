import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class Config:
  TRAIN_DIR       : str = "train dataset path"
  VAL_DIR         : str = "validation dataset path"
  TEST_DIR        : str = "test dataset path"

  SAVE_PATH   : str = 'save path(log)'
  OUTPUT_PATH : str = 'resual path(visual)'
  VALID_EXT       : Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')


  # ── Reproducibility ─────────────────────────────────────────────
  SEED       : int          = 42
  DEVICE     : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  EXPERIMENT : str          = 'ConvNeXt_AutoEncoder_Anomaly'
  # ── Label keywords (filename-based) ─────────────────────────────
  NORMAL_KEYWORDS  : Tuple[str, ...] = ('false_call','falsecall','good','normal','false call')
  ANOMALY_KEYWORDS : Tuple[str, ...] = ('defect','anomaly','bad','ng')
  # ── ConvNeXt backbone ───────────────────────────────────────────
  LOSS         : str            = 'MSE'
  SSIM_WEIGHT  : float          = 0.5
  MSE_WEIGHT   : float          = 0.5
  OPTIM        : str            = 'Adam (Adaptive Moment Estimation)'
  BACKBONE     : str            = 'tiny'
  IMAGE_SIZE   : Tuple[int,int] = (224, 224)
  # ── DataLoader ──────────────────────────────────────────────────
  BATCH_SIZE   : int  = 64
  NUM_WORKERS  : int  = 2
  PIN_MEMORY   : bool = True

  # ── Autoencoder Training ─────────────────────────────────────────
  AE_EPOCHS         : int   = 150
  AE_LR             : float = 1e-4
  AE_WEIGHT_DECAY   : float = 5e-4
  AE_BOTTLENECK_CH  : int   = 64
  AE_PATIENCE       : int   = 10
  AE_LR_STEP        : int   = 15
  AE_LR_GAMMA       : float = 0.5

  # ── Heatmap ─────────────────────────────────────────────────────
  HEATMAP_SIGMA         : float = 4.0
  THRESHOLD_PERCENTILE  : float = 95.0

  SCORE_METHOD          : str   = 'topk'
  SCORE_TOPK_PERCENT    : float = 10.0
  AE_MONITOR            : str   = 'val_auroc'
  USE_AUGMENTATION      : bool  = False
  AUG_COLOR_JITTER      : float = 0

  # ── Preprocessing / Color Mode ──────────────────────────────────────
  # เลือกโหมดสีของภาพก่อนเข้า pipeline ด้วยการตั้งค่า True/False 2 ตัวนี้:
  #
  #   โหมด RGB (ค่า default, ไม่แปลงสี)
  #     USE_GRAYSCALE = False, USE_GRAYSCALE_EQUALIZATION = False
  #
  #   โหมด Grayscale (แปลงเป็นขาวดำ ไม่ equalize)
  #     USE_GRAYSCALE = True,  USE_GRAYSCALE_EQUALIZATION = False
  #
  #   โหมด Grayscale + Histogram Equalization (แปลงขาวดำ + เพิ่ม contrast)
  #     USE_GRAYSCALE_EQUALIZATION = True
  #     (ตั้ง USE_GRAYSCALE เป็นค่าใดก็ได้ — equalization บังคับใช้ grayscale
  #      อยู่แล้วในตัว จึงมีความสำคัญเหนือกว่า USE_GRAYSCALE เสมอ)
  #
  # หมายเหตุ: เลือกได้ทีละโหมดเท่านั้น ถ้า USE_GRAYSCALE_EQUALIZATION=True
  # ระบบจะใช้โหมด grayscale+equalize เสมอ ไม่ว่า USE_GRAYSCALE จะเป็นอะไร
  USE_GRAYSCALE               : bool = False
  USE_GRAYSCALE_EQUALIZATION  : bool = False

  @property
  def COLOR_MODE(self) -> str:
    """โหมดสีที่ระบบจะใช้จริง (คำนวณจากแฟล็กทั้งสองด้านบน)."""
    if self.USE_GRAYSCALE_EQUALIZATION:
      return 'grayscale_equalized'
    elif self.USE_GRAYSCALE:
      return 'grayscale'
    else:
      return 'rgb'

  # ── Preprocessing / Color Mode ──────────────────────────────────────
  # เลือกโหมดสีของภาพก่อนเข้า pipeline ด้วยการตั้งค่า True/False 2 ตัวนี้:
  #
  #   โหมด RGB (ค่า default, ไม่แปลงสี)
  #     USE_GRAYSCALE = False, USE_GRAYSCALE_EQUALIZATION = False
  #
  #   โหมด Grayscale (แปลงเป็นขาวดำ ไม่ equalize)
  #     USE_GRAYSCALE = True,  USE_GRAYSCALE_EQUALIZATION = False
  #
  #   โหมด Grayscale + Histogram Equalization (แปลงขาวดำ + เพิ่ม contrast)
  #     USE_GRAYSCALE_EQUALIZATION = True
  #     (ตั้ง USE_GRAYSCALE เป็นค่าใดก็ได้ — equalization บังคับใช้ grayscale
  #      อยู่แล้วในตัว จึงมีความสำคัญเหนือกว่า USE_GRAYSCALE เสมอ)
  #
  # หมายเหตุ: เลือกได้ทีละโหมดเท่านั้น ถ้า USE_GRAYSCALE_EQUALIZATION=True
  # ระบบจะใช้โหมด grayscale+equalize เสมอ ไม่ว่า USE_GRAYSCALE จะเป็นอะไร
  USE_GRAYSCALE               : bool = False
  USE_GRAYSCALE_EQUALIZATION  : bool = False

  @property
  def COLOR_MODE(self) -> str:
    """โหมดสีที่ระบบจะใช้จริง (คำนวณจากแฟล็กทั้งสองด้านบน)."""
    if self.USE_GRAYSCALE_EQUALIZATION:
      return 'grayscale_equalized'
    elif self.USE_GRAYSCALE:
      return 'grayscale'
    else:
      return 'rgb'

  def __post_init__(self):
    for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
      Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
