import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class Config:
  TRAIN_DIR       : str = "/content/drive/MyDrive/Thesis/DATASET/ชุดที่3/Anomaly3/train_anomaly3"
  VAL_DIR         : str = "/content/drive/MyDrive/Thesis/DATASET/ชุดที่3/Anomaly3/val_anomaly3"
  TEST_DIR        : str = "/content/drive/MyDrive/Thesis/DATASET/ชุดที่3/Anomaly3/test_anomaly3"

  SAVE_PATH   : str = '/content/drive/MyDrive/Thesis/EXPERIMENT/EXPERIMENT4(ConvNeXt_Autoencoder)/SAVED/ConvNeXt_AE_3_1(Top-K)'
  OUTPUT_PATH : str = '/content/drive/MyDrive/Thesis/EXPERIMENT/EXPERIMENT4(ConvNeXt_Autoencoder)/OUTPUT/ConvNeXt_AE_3_1(Top-K)'
  VALID_EXT       : Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')


  # ── Reproducibility ─────────────────────────────────────────────
  SEED       : int          = 42
  DEVICE     : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  EXPERIMENT : str          = 'ConvNeXt_AutoEncoder_Anomaly'
  # ── Label keywords (filename-based) ─────────────────────────────
  NORMAL_KEYWORDS  : Tuple[str, ...] = ('false_call','falsecall','good','normal','false call')
  ANOMALY_KEYWORDS : Tuple[str, ...] = ('defect','anomaly','bad','ng')
  # ── ConvNeXt backbone ───────────────────────────────────────────
  LOSS         : str            = 'SSIM_MSE'
  SSIM_WEIGHT  : float          = 0.5
  MSE_WEIGHT   : float          = 0.5
  OPTIM        : str            = 'Adam (Adaptive Moment Estimation)'
  BACKBONE     : str            = 'tiny'
  IMAGE_SIZE   : Tuple[int,int] = (224, 224)
  # ── DataLoader ──────────────────────────────────────────────────
  BATCH_SIZE   : int  = 16
  NUM_WORKERS  : int  = 2
  PIN_MEMORY   : bool = True

  # ── Autoencoder Training ─────────────────────────────────────────
  AE_EPOCHS         : int   = 150
  AE_LR             : float = 1e-4
  AE_WEIGHT_DECAY   : float = 5e-4
  AE_BOTTLENECK_CH  : int   = 64
  AE_PATIENCE       : int   = 15
  AE_LR_STEP        : int   = 15
  AE_LR_GAMMA       : float = 0.5

  # ── Heatmap ─────────────────────────────────────────────────────
  HEATMAP_SIGMA         : float = 4.0
  THRESHOLD_PERCENTILE  : float = 95.0

  SCORE_METHOD          : str   = 'topk'
  SCORE_TOPK_PERCENT    : float = 10.0
  AE_MONITOR            : str   = 'val_auroc'
  USE_AUGMENTATION      : bool  = True
  AUG_ROTATION_DEG      : float = 10.0
  AUG_TRANSLATE         : float = 0.05
  AUG_COLOR_JITTER      : float = 0.20

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
