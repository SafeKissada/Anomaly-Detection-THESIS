"""Save/load helpers for all artifacts produced by scripts/train.py and
consumed by scripts/visualize.py:

- history.json               (training history)
- extractor_norm_stats.pt    (feature mean/std fitted on normal-only data)
- scores_{split}.npz         (scores, y_true, paths, labels, heatmaps, orig_imgs)
- threshold.json             (deployment threshold + oracle diagnostic)
- best_autoencoder.pth / autoencoder_final.pth (checkpoints)
"""

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
SPLITS          = ('train', 'val', 'test')


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
  """Restore fitted normalization stats into an extractor (no re-fitting)."""
  stats = load_norm_stats(cfg, map_location=map_location)
  extractor.feat_mean = stats['feat_mean'].to(extractor.feat_mean.device)
  extractor.feat_std  = stats['feat_std'].to(extractor.feat_std.device)
  extractor.norm_fitted = True


# ── scores_{split}.npz ───────────────────────────────────────────────────────

def save_scores(split: str, scores: np.ndarray, y_true: np.ndarray,
                paths: List[str], labels: List[str],
                heatmaps: List[np.ndarray], orig_imgs: List[np.ndarray],
                cfg) -> str:
  path = scores_path(split, cfg)
  np.savez_compressed(
      path,
      scores    = np.asarray(scores, dtype=np.float32),
      y_true    = np.asarray(y_true, dtype=int),
      paths     = np.array(paths),
      labels    = np.array(labels),
      heatmaps  = np.stack(heatmaps).astype(np.float32),
      orig_imgs = np.stack(orig_imgs).astype(np.float32),
  )
  return path


def load_scores(split: str, cfg) -> Dict:
  with np.load(scores_path(split, cfg)) as data:
    return {
        'scores'    : data['scores'],
        'y_true'    : data['y_true'],
        'paths'     : [str(p) for p in data['paths']],
        'labels'    : [str(l) for l in data['labels']],
        'heatmaps'  : list(data['heatmaps']),
        'orig_imgs' : list(data['orig_imgs']),
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


# ── artifact presence check ──────────────────────────────────────────────────

def require_artifacts(cfg, splits=SPLITS) -> None:
  """Raise FileNotFoundError if any artifact needed by visualize.py is missing."""
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
