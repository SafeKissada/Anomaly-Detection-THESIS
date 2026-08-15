"""Config dataclass เดียวสำหรับทั้ง pipeline — ทุก hyperparameter อยู่ที่นี่
ที่เดียว พร้อม validation แบบ fail-fast ใน __post_init__ (ผิด → error ทันที
ตอนสร้าง Config() ไม่ปล่อยให้ไปพังกลางทางตอนเทรน)

Single Config dataclass for the whole pipeline — every hyperparameter
lives here, with fail-fast validation in __post_init__ (a mistake raises
immediately at Config() construction instead of failing mid-training).
"""
import random
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

  # regex ที่มี capture group 1 ตัว ใช้กัน sample ชุดเดียวกัน (เช่น ภาพจาก
  # ผลิตภัณฑ์ตัวเดียวกันหลายมุม) หลุดไปอยู่คนละ split — None = ปิดการกันนี้
  # A regex with exactly one capture group, used to keep samples from the
  # same physical group (e.g. multiple angles of one product) from
  # landing in different splits — None disables this grouping check.
  GROUP_ID_REGEX : Optional[str] = None

  SAVE_PATH   : str = 'save log'
  OUTPUT_PATH : str = 'save image/table'
  VALID_EXT       : Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')


  # ── Reproducibility / ทำซ้ำผลได้ ─────────────────────────────────
  SEED       : int          = 42
  DEVICE     : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  EXPERIMENT : str          = 'EXPERIMENT --  --'
  # ── Loss & optimizer ───────────────────────────────────────────
  LOSS         : str            = 'MSE'
  # Weight ของ cosine term ใน CosineMSELoss (LOSS='COS_MSE'):
  #   loss = COS_LAM * cosine_distance + (1 - COS_LAM) * mse
  # lam=1.0 = pure cosine, lam=0.0 = pure MSE. ไม่ถูกใช้เมื่อ LOSS อื่น
  # (รวมถึง LOSS='COS' ซึ่งเป็น pure cosine เสมอ ไม่มี weight ให้ปรับ)
  #
  # Weight of the cosine term in CosineMSELoss (LOSS='COS_MSE'):
  #   loss = COS_LAM * cosine_distance + (1 - COS_LAM) * mse
  # lam=1.0 = pure cosine, lam=0.0 = pure MSE. Not read by any other LOSS
  # (including LOSS='COS', which is always pure cosine — no weight to tune).
  COS_LAM      : float          = 0.5
  # Stability epsilon ของ F.cosine_similarity (ป้องกันหารด้วยศูนย์เมื่อ
  # feature vector ที่ตำแหน่งใดตำแหน่งหนึ่งมี norm ใกล้ 0). ใช้เมื่อ
  # LOSS='COS'/'COS_MSE'. ปกติไม่ต้องปรับ แต่เปิดให้ ablate ได้เผื่อจำเป็น.
  #
  # Stability epsilon for F.cosine_similarity (guards against division by
  # zero when a feature vector at some position has near-zero norm). Used
  # when LOSS='COS'/'COS_MSE'. Normally doesn't need tuning, exposed here
  # in case an ablation needs it.
  COS_EPS          : float       = 1e-8
  # ค่า threshold (delta) ที่ nn.HuberLoss สลับจาก quadratic (เหมือน MSE,
  # ตอน |error| < delta) ไปเป็น linear (เหมือน MAE, ตอน |error| >= delta)
  # ใช้เมื่อ cfg.LOSS เป็น 'HUBER'/'SMOOTH_L1' เท่านั้น
  #
  # Threshold (delta) at which nn.HuberLoss switches from quadratic (like
  # MSE, for |error| < delta) to linear (like MAE, for |error| >= delta)
  # behavior. Only used when cfg.LOSS is 'HUBER'/'SMOOTH_L1'.
  HUBER_DELTA  : float          = 1.0
  # optimizer ที่ src/optim.get_optimizer() จะสร้างให้ autoencoder
  # ค่าที่รองรับ: 'Adam' | 'AdamW' | 'SGD' | 'RMSprop' (ไม่สนตัวพิมพ์เล็ก-ใหญ่)
  #
  # Which optimizer src/optim.get_optimizer() builds for the autoencoder.
  # One of 'Adam' | 'AdamW' | 'SGD' | 'RMSprop' (case-insensitive).
  OPTIM        : str            = 'Adam'
  # Momentum — ใช้เฉพาะ SGD กับ RMSprop เท่านั้น (Adam/AdamW ไม่สนค่านี้
  # เพราะใช้ beta1/beta2 ภายในของตัวเองแทน)
  #
  # Momentum — only read by SGD and RMSprop (Adam/AdamW ignore it and use
  # their own internal beta1/beta2 instead).
  AE_MOMENTUM      : float = 0.9
  # Nesterov momentum — เฉพาะ SGD เท่านั้น และมีผลก็ต่อเมื่อ AE_MOMENTUM > 0
  # (torch.optim.SGD บังคับว่า momentum ต้อง > 0 ถึงจะใช้ nesterov=True ได้
  # get_optimizer() จะบังคับตั้งเป็น False ให้อัตโนมัติแทนการ raise error)
  #
  # Nesterov momentum — SGD only, and only takes effect if AE_MOMENTUM > 0
  # (torch.optim.SGD requires momentum > 0 for nesterov=True; get_optimizer()
  # forces this False automatically otherwise rather than raising).
  AE_SGD_NESTEROV  : bool  = True
  # ค่าเฉพาะของ RMSprop เท่านั้น / RMSprop-only knobs.
  AE_RMSPROP_ALPHA : float = 0.99
  AE_RMSPROP_EPS   : float = 1e-8
  BACKBONE     : str            = 'tiny'
  IMAGE_SIZE   : Tuple[int,int] = (224, 224)
  # ── DataLoader ──────────────────────────────────────────────────
  BATCH_SIZE   : int  = 32
  NUM_WORKERS  : int  = 2
  PIN_MEMORY   : bool = True

  # ── Autoencoder Training / การเทรน autoencoder ────────────────────
  AE_EPOCHS         : int   = 100
  AE_LR             : float = 1e-4
  AE_WEIGHT_DECAY   : float = 5e-4
  AE_BOTTLENECK_CH  : int   = 64
  AE_LR_STEP        : int   = 25
  AE_LR_GAMMA       : float = 0.5
  AE_PATIENCE       : int   = 20

  # ── Heatmap & scoring ───────────────────────────────────────────
  HEATMAP_SIGMA         : float = 4.0
  THRESHOLD_PERCENTILE  : float = 95.0

  SCORE_METHOD          : str   = 'topk'
  SCORE_TOPK_PERCENT    : float = 10.0
  AE_MONITOR            : str   = 'val_loss'
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
  # หมายเหตุ: เลือกได้ทีละโหมด ถ้า USE_GRAYSCALE_EQUALIZATION=True
  # ระบบจะใช้โหมด grayscale+equalize เสมอ ไม่ว่า USE_GRAYSCALE จะเป็นอะไร
  #
  # Select the image color mode before the pipeline via these two
  # True/False flags:
  #
  #   RGB mode (default, no color conversion)
  #     USE_GRAYSCALE = False, USE_GRAYSCALE_EQUALIZATION = False
  #
  #   Grayscale mode (converted to grayscale, not equalized)
  #     USE_GRAYSCALE = True,  USE_GRAYSCALE_EQUALIZATION = False
  #
  #   Grayscale + Histogram Equalization (grayscale + boosted contrast)
  #     USE_GRAYSCALE_EQUALIZATION = True
  #     (USE_GRAYSCALE can be anything — equalization implies grayscale
  #      on its own, so it always takes priority over USE_GRAYSCALE)
  #
  # Note: only one mode applies at a time. If USE_GRAYSCALE_EQUALIZATION=
  # True, the system always uses grayscale+equalize regardless of
  # USE_GRAYSCALE.
  USE_GRAYSCALE               : bool = False
  USE_GRAYSCALE_EQUALIZATION  : bool = False
  USE_CLAHE                   : bool = False

  # ค่า hyperparameter ของ CLAHE (ใช้เมื่อ USE_CLAHE=True เท่านั้น)
  # ตรงกับ cv2.createCLAHE(clipLimit=..., tileGridSize=...)
  #
  # CLAHE hyperparameters (used only when USE_CLAHE=True), matching
  # cv2.createCLAHE(clipLimit=..., tileGridSize=...)
  CLAHE_CLIP_LIMIT       : float = 2.0
  CLAHE_TILE_GRID_SIZE   : tuple = (8, 8)

  @property
  def COLOR_MODE(self) -> str:
    """โหมดสีที่ derive มาจาก flag ด้านบน (ไม่ใช่ field ที่ตั้งตรงๆ)

    ลำดับความสำคัญ (ต้องตรงกับ build_transforms() ใน src/data/dataset.py):
      USE_GRAYSCALE_EQUALIZATION และ USE_CLAHE -> GRAYSCALE_EQUALIZATION_CLAHE
      USE_GRAYSCALE_EQUALIZATION อย่างเดียว     -> GRAYSCALE_EQUALIZATION
      USE_CLAHE อย่างเดียว                      -> GRAYSCALE_CLAHE
      USE_GRAYSCALE อย่างเดียว                  -> GRAYSCALE
      ไม่ตั้งอะไรเลย                             -> RGB

    Derived color mode from the flags above (not a directly-set field).

    Priority (matches build_transforms() in src/data/dataset.py):
      USE_GRAYSCALE_EQUALIZATION and USE_CLAHE -> GRAYSCALE_EQUALIZATION_CLAHE
      USE_GRAYSCALE_EQUALIZATION only          -> GRAYSCALE_EQUALIZATION
      USE_CLAHE only                           -> GRAYSCALE_CLAHE
      USE_GRAYSCALE only                       -> GRAYSCALE
      none set                                 -> RGB
    """
    if self.USE_GRAYSCALE_EQUALIZATION and self.USE_CLAHE:
      return 'GRAYSCALE_EQUALIZATION_CLAHE'
    elif self.USE_GRAYSCALE_EQUALIZATION:
      return 'GRAYSCALE_EQUALIZATION'
    elif self.USE_CLAHE:
      return 'GRAYSCALE_CLAHE'
    elif self.USE_GRAYSCALE:
      return 'GRAYSCALE'
    else:
      return 'RGB'

  _DATA_ROOT_PLACEHOLDER = "dataset root path (contains good/ and defect/ subfolders)"

  _VALID_OPTIMS = ('ADAM', 'ADAMW', 'SGD', 'RMSPROP')

  # ต้องตรงกับทุก alias ที่ src/losses.py:get_criterion() รู้จักจริง —
  # ถ้าเพิ่ม loss ใหม่ใน get_criterion() ต้องเพิ่มที่นี่ด้วย ไม่งั้น Config()
  # จะ reject ค่าที่ get_criterion() รองรับอยู่แล้ว
  #
  # Must match every alias src/losses.py:get_criterion() actually
  # recognizes — adding a new loss to get_criterion() requires adding it
  # here too, or Config() will reject a value get_criterion() supports.
  _VALID_LOSSES = ('MSE', 'MAE', 'L1', 'HUBER', 'SMOOTH_L1', 'SMOOTHL1',
                    'COS', 'COS_MSE', 'COS+MSE', 'COSMSE')

  def __post_init__(self):
    for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
      Path(p).mkdir(parents=True, exist_ok=True)

    if self.OPTIM.strip().upper() not in self._VALID_OPTIMS:
      raise ValueError(
          f"Config.OPTIM must be one of {self._VALID_OPTIMS} "
          f"(case-insensitive), got {self.OPTIM!r}.")

    if self.LOSS.strip().upper() not in self._VALID_LOSSES:
      raise ValueError(
          f"Config.LOSS must be one of {self._VALID_LOSSES} "
          f"(case-insensitive), got {self.LOSS!r}. This is checked eagerly "
          f"here (fail-fast, like Config.OPTIM) instead of letting a typo "
          f"propagate silently until src/losses.py:get_criterion() raises "
          f"mid-training in scripts/train.py.")

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

    # Fail fast ตอนสร้าง Config() เลย พร้อม error message ที่แก้ปัญหาได้จริง —
    # แทนที่จะปล่อยให้ placeholder string หลุดรอดลงไปถึง _list_labeled_files()
    # แล้วพังที่นั่นด้วย path ที่งงๆ (เช่น ".../dataset root path (contains
    # good/ and defect/ subfolders)/good") ตั้งใจไม่ครอบ try/except ตรงไหน
    # ใน call chain เลย เพราะ DATA_ROOT ตั้งผิดคือ setup mistake ที่ต้องแก้
    # ที่ config.py ไม่ใช่ runtime condition ที่ retry หรือกลบเกลื่อนได้
    #
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
  """ตั้ง seed ให้ทุก RNG ที่เกี่ยวข้อง (random, numpy, torch, cuda) เพื่อให้
  ผลรันซ้ำได้ — เรียกครั้งเดียวตอนต้นของ main() ก่อนสร้าง dataset/model ใดๆ

  Seed every relevant RNG (random, numpy, torch, cuda) for reproducible
  runs — call once at the start of main(), before creating any
  dataset/model.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False