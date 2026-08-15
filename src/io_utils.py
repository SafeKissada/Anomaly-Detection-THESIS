"""ฟังก์ชัน save/load สำหรับทุก artifact ที่ scripts/train.py สร้างและ
scripts/visualize.py อ่านต่อ:

- history.json               (training history)
- extractor_norm_stats.pt    (feature mean/std ที่ fit จากข้อมูล normal-only)
- scores_{split}.npz         (scores, y_true, paths, labels, heatmaps, orig_imgs)
- threshold.json             (deployment threshold + oracle diagnostic)
- best_autoencoder.pth / autoencoder_final.pth (checkpoint)

มี config_to_serializable_dict() ให้ด้วย ซึ่ง scripts/train.py ใช้ฝัง
snapshot เต็มของทุก field ใน Config ลงใน final_results.json ทำให้ไฟล์
ผลลัพธ์ของแต่ละ experiment อธิบายตัวเองได้ครบ (จำเป็นสำหรับแยกแยะ E0 vs E1
vs E2 ฯลฯ ใน ablation study โดยไม่ต้องย้อนไปดู config.py)

Save/load helpers for all artifacts produced by scripts/train.py and
consumed by scripts/visualize.py:

- history.json               (training history)
- extractor_norm_stats.pt    (feature mean/std fitted on normal-only data)
- scores_{split}.npz         (scores, y_true, paths, labels, heatmaps, orig_imgs)
- threshold.json             (deployment threshold + oracle diagnostic)
- best_autoencoder.pth / autoencoder_final.pth (checkpoints)

Also provides config_to_serializable_dict(), used by scripts/train.py to
embed a full snapshot of every Config field into final_results.json, so
each experiment's results file is self-documenting (needed to tell E0 vs E1
vs E2 etc. apart in an ablation study without cross-referencing config.py).
"""

import dataclasses
import json
import os
from typing import Dict, List

import numpy as np
import torch

HISTORY_FILE    = 'history.json'
NORM_STATS_FILE = 'extractor_norm_stats.pt'
THRESHOLD_FILE  = 'threshold.json'
BEST_CKPT_FILE  = 'best_autoencoder.pth'
FINAL_CKPT_FILE = 'autoencoder_final.pth'
SPLITS          = ('val', 'test')


def history_path(cfg) -> str:
  return os.path.join(cfg.SAVE_PATH, HISTORY_FILE)


def norm_stats_path(cfg) -> str:
  return os.path.join(cfg.SAVE_PATH, NORM_STATS_FILE)


def threshold_path(cfg) -> str:
  return os.path.join(cfg.SAVE_PATH, THRESHOLD_FILE)


def scores_path(split: str, cfg) -> str:
  return os.path.join(cfg.SAVE_PATH, f'scores_{split}.npz')


def checkpoint_path(cfg, name: str = BEST_CKPT_FILE) -> str:
  return os.path.join(cfg.SAVE_PATH, name)


# ── full config snapshot (for final_results.json) / snapshot ของ config เต็ม ──

def config_to_serializable_dict(cfg) -> Dict:
  """แปลง instance ของ Config dataclass เป็น dict ธรรมดาที่ serialize เป็น
  JSON ได้ ครอบคลุม**ทุก** field ที่ตั้งค่าไว้ (LOSS, HUBER_DELTA, COS_LAM,
  COS_EPS, ทุก hyperparameter AE_*, SCORE_METHOD, SPLIT_RATIOS, SEED, โหมดสี
  ฯลฯ) — ไม่ใช่แค่ subset เล็กๆ ที่เลือกมือที่ scripts/train.py เคยใช้ใส่
  final_results.json ก่อนหน้านี้

  ตั้งใจให้ฝังไว้ใต้ summary_dict['config'] ใน final_results.json เพื่อให้
  แต่ละ run ของ experiment (เช่น แยก E0 vs E1 vs E2 ใน ablation study)
  อธิบายตัวเองได้ครบจากไฟล์ผลลัพธ์ของมันเอง โดยไม่ต้องย้อนไปหาว่า
  config.py ตอนนั้นตั้งค่าอะไรไว้

  มี 2 เรื่องที่ dataclasses.asdict() เพียวๆ จะพลาด/พังถ้าไม่จัดการเพิ่ม:
    - COLOR_MODE เป็น @property ไม่ใช่ dataclass field ธรรมดา asdict()
      จึงไม่รวมให้อัตโนมัติ — เพิ่มเข้ามาตรงๆ ที่นี่ เพราะมันถูกโชว์ใน
      console panel ตอนเทรน (scripts/train.py) และเป็นค่าที่ derive มา
      ซึ่งมีประโยชน์ที่จะบันทึกไว้ (RGB / GRAYSCALE / GRAYSCALE_
      EQUALIZATION)
    - DEVICE เป็น object ชนิด torch.device ซึ่ง json.dump() serialize เอง
      ไม่ได้ — แปลงเป็น string ก่อน (เช่น 'cuda' หรือ 'cpu')

  Convert a Config dataclass instance into a plain, JSON-serializable
  dict containing EVERY configured field (LOSS, HUBER_DELTA, COS_LAM,
  COS_EPS, all AE_* hyperparameters, SCORE_METHOD, SPLIT_RATIOS, SEED,
  color mode, etc.) — not just the small hand-picked subset scripts/train.py
  used to put in final_results.json before this.

  This is meant to be embedded under summary_dict['config'] in
  final_results.json so that each experiment run (e.g. distinguishing E0 vs
  E1 vs E2 in an ablation study) is fully self-documented by its own results
  file — no need to separately track down which values config.py held at
  the time that particular experiment was run.

  Two things dataclasses.asdict() would otherwise miss/break on:
    - COLOR_MODE is a @property, not a dataclass field, so asdict() does
      not include it — added explicitly here since it's shown in the
      training-time console panel (scripts/train.py) and is a genuinely
      derived, useful-to-record setting (RGB / GRAYSCALE / GRAYSCALE_
      EQUALIZATION).
    - DEVICE is a torch.device object, which json.dump() cannot serialize
      on its own — converted to its string form (e.g. 'cuda' or 'cpu').
  """
  d = dataclasses.asdict(cfg)
  d['DEVICE'] = str(cfg.DEVICE)
  d['COLOR_MODE'] = cfg.COLOR_MODE
  return d


# ── history.json ─────────────────────────────────────────────────────────────

def save_history(history: Dict, cfg) -> str:
  path = history_path(cfg)
  serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
  with open(path, 'w') as f:
    json.dump(serializable, f, indent=2)
  return path


def load_history(cfg) -> Dict:
  with open(history_path(cfg)) as f:
    return json.load(f)


# ── extractor_norm_stats.pt ──────────────────────────────────────────────────

def save_norm_stats(extractor, cfg) -> str:
  path = norm_stats_path(cfg)
  torch.save({
      'feat_mean'    : extractor.feat_mean.detach().cpu(),
      'feat_std'     : extractor.feat_std.detach().cpu(),
      'out_channels' : extractor.out_channels,
      'spatial_size' : tuple(extractor.spatial_size),
  }, path)
  return path


def load_norm_stats(cfg, map_location='cpu') -> Dict:
  return torch.load(norm_stats_path(cfg), map_location=map_location)


def load_norm_stats_into(extractor, cfg, map_location='cpu') -> None:
  """คืนค่าสถิติ normalize ที่ fit ไว้แล้วกลับเข้า extractor (ไม่ fit ใหม่)

  Restore fitted normalization stats into an extractor (no re-fitting).
  """
  stats = load_norm_stats(cfg, map_location=map_location)
  extractor.feat_mean = stats['feat_mean'].to(extractor.feat_mean.device)
  extractor.feat_std  = stats['feat_std'].to(extractor.feat_std.device)
  extractor.norm_fitted = True


# ── scores_{split}.npz ───────────────────────────────────────────────────────

def save_scores(split: str, scores: np.ndarray, y_true: np.ndarray,
                paths: List[str], labels: List[str],
                heatmaps: List[np.ndarray], orig_imgs: List[np.ndarray],
                cfg, preproc_imgs: List[np.ndarray] = None) -> str:
  path = scores_path(split, cfg)
  # preproc_imgs อาจไม่ถูกส่งมาก็ได้ เพื่อ backward compatibility; ถ้าไม่มี
  # ใช้ orig_imgs แทน (เกิดขึ้นเองตามธรรมชาติในโหมด RGB อยู่แล้ว เพราะสอง
  # อย่างนี้เหมือนกันเป๊ะ)
  #
  # preproc_imgs may be omitted for backward compatibility; default to orig_imgs
  # (this is also what happens naturally in RGB mode, where they're identical).
  if preproc_imgs is None:
    preproc_imgs = orig_imgs
  np.savez_compressed(
      path,
      scores       = np.asarray(scores, dtype=np.float32),
      y_true       = np.asarray(y_true, dtype=int),
      paths        = np.array(paths),
      labels       = np.array(labels),
      heatmaps     = np.stack(heatmaps).astype(np.float32),
      orig_imgs    = np.stack(orig_imgs).astype(np.float32),
      preproc_imgs = np.stack(preproc_imgs).astype(np.float32),
  )
  return path


def load_scores(split: str, cfg) -> Dict:
  with np.load(scores_path(split, cfg)) as data:
    return {
        'scores'      : data['scores'],
        'y_true'      : data['y_true'],
        'paths'       : [str(p) for p in data['paths']],
        'labels'      : [str(l) for l in data['labels']],
        'heatmaps'    : list(data['heatmaps']),
        'orig_imgs'   : list(data['orig_imgs']),
        # 'preproc_imgs' อาจไม่มีใน .npz ไฟล์ที่เซฟไว้ก่อนจะเพิ่ม field นี้
        # เข้ามา — fallback ไปใช้ orig_imgs แทน (เทียบเท่ากับโหมด RGB)
        #
        # 'preproc_imgs' may be missing in .npz files saved before this field
        # was added — fall back to orig_imgs (equivalent to RGB mode).
        'preproc_imgs': list(data['preproc_imgs']) if 'preproc_imgs' in data
                        else list(data['orig_imgs']),
    }


# ── threshold.json ───────────────────────────────────────────────────────────

def save_threshold(threshold: float, percentile: float,
                   oracle_threshold: float, oracle_f1: float, cfg) -> str:
  path = threshold_path(cfg)
  with open(path, 'w') as f:
    json.dump({
        'threshold'        : float(threshold),
        'percentile'       : float(percentile),
        'oracle_threshold' : float(oracle_threshold),
        'oracle_f1'        : float(oracle_f1),
    }, f, indent=2)
  return path


def load_threshold(cfg) -> Dict:
  with open(threshold_path(cfg)) as f:
    return json.load(f)


# ── artifact presence check / เช็คว่า artifact ครบไหม ──────────────────────

def require_artifacts(cfg, splits=SPLITS) -> None:
  """Raise FileNotFoundError ถ้ามี artifact ตัวไหนที่ visualize.py ต้องใช้หายไป

  Raise FileNotFoundError if any artifact needed by visualize.py is missing.
  """
  required = [
      history_path(cfg),
      norm_stats_path(cfg),
      threshold_path(cfg),
      checkpoint_path(cfg, BEST_CKPT_FILE),
      checkpoint_path(cfg, FINAL_CKPT_FILE),
  ] + [scores_path(s, cfg) for s in splits]

  missing = [p for p in required if not os.path.exists(p)]
  if missing:
    missing_str = '\n  '.join(missing)
    raise FileNotFoundError(
        f'Missing required artifact(s):\n  {missing_str}\n'
        f'Run scripts/train.py first to generate them.')