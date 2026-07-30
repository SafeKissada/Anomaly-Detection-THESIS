import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
 
import numpy as np
import torch
 
 
@dataclass
class Config:
  DATA_ROOT     : str = "dataset root path (contains good/ and defect/ subfolders)"
  GOOD_DIRNAME  : str = "good"
  DEFECT_DIRNAME: str = "defect"
 
  SPLIT_RATIOS  : Tuple[float, float, float] = (0.70, 0.15, 0.15)
 
  SPLIT_CACHE_PATH : str = "splits/split_assignment.csv"
 
  GROUP_ID_REGEX : Optional[str] = None
 
  SAVE_PATH   : str = 'save log'
  OUTPUT_PATH : str = 'save image/table'
  VALID_EXT       : Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
 
 
  # ── Reproducibility ─────────────────────────────────────────────
  SEED       : int          = 42
  DEVICE     : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  EXPERIMENT : str          = 'ConvNeXt_AutoEncoder_Anomaly'
  # ── ConvNeXt backbone ───────────────────────────────────────────
  LOSS         : str            = 'MSE'
  SSIM_WEIGHT  : float          = 0.5
  MSE_WEIGHT   : float          = 0.5
  OPTIM        : str            = 'Adam'
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
  AUG_COLOR_JITTER      : float = 0.20
 
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
  # Note: เลือกได้ทีละโหมด ถ้า USE_GRAYSCALE_EQUALIZATION=True
  # ระบบจะใช้โหมด grayscale+equalize เสมอ ไม่ว่า USE_GRAYSCALE จะเป็นอะไร
  USE_GRAYSCALE               : bool = False
  USE_GRAYSCALE_EQUALIZATION  : bool = False
 
  @property
  def COLOR_MODE(self) -> str:
    """โหมดปรับ Image Processing."""
    if self.USE_GRAYSCALE_EQUALIZATION:
      return 'GRAYSCALE_EQUALIZATION'
    elif self.USE_GRAYSCALE:
      return 'GRAYSCALE'
    else:
      return 'RGB'
 
  _DATA_ROOT_PLACEHOLDER = "dataset root path (contains good/ and defect/ subfolders)"
 
  def __post_init__(self):
    for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
      Path(p).mkdir(parents=True, exist_ok=True)
 
    ratio_sum = sum(self.SPLIT_RATIOS)
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
      raise ValueError(
          f"Config.SPLIT_RATIOS must sum to 1.0, got {self.SPLIT_RATIOS} "
          f"(sums to {ratio_sum}). This is checked eagerly here rather than "
          f"left to silently produce a smaller-or-overlapping split later.")
    if len(self.SPLIT_RATIOS) != 3:
      raise ValueError(
          f"Config.SPLIT_RATIOS must have exactly 3 values (train, val, "
          f"test), got {len(self.SPLIT_RATIOS)}: {self.SPLIT_RATIOS}")
 
    # Fail fast, at Config() construction time, with an actionable message —
    # instead of letting the placeholder string silently propagate all the
    # way down into _list_labeled_files() and fail there with a path that
    # looks confusing (e.g. ".../dataset root path (contains good/ and
    # defect/ subfolders)/good"). This is deliberately NOT wrapped in a
    # try/except anywhere in the call chain: a misconfigured DATA_ROOT is a
    # setup mistake that must be fixed by editing config.py, not a
    # recoverable runtime condition to retry or paper over.
    if self.DATA_ROOT == self._DATA_ROOT_PLACEHOLDER:
      raise ValueError(
          "Config.DATA_ROOT is still the default placeholder string. Set it "
          "to a real folder on your machine that contains two subfolders "
          f"named cfg.GOOD_DIRNAME ({self.GOOD_DIRNAME!r}) and "
          f"cfg.DEFECT_DIRNAME ({self.DEFECT_DIRNAME!r}), e.g.:\n"
          '    DATA_ROOT : str = "C:/path/to/your/dataset"  (Windows: use '
          "forward slashes or an r'...' raw string)\n"
          '    DATA_ROOT : str = "/path/to/your/dataset"    (Linux/Mac)')
    if not Path(self.DATA_ROOT).is_dir():
      raise FileNotFoundError(
          f"Config.DATA_ROOT does not exist or is not a directory: "
          f"{self.DATA_ROOT!r}. Double-check the path (and, on Windows, "
          f"that backslashes are either doubled '\\\\' or written as an "
          f"r'...' raw string / forward slashes).")
 
 
def set_seed(seed: int = 42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
