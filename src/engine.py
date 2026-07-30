import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.losses import SSIMLoss, CombinedLoss, get_criterion, elementwise_error_map


class EarlyStopping:
  def __init__(self,
               patience: int   = 15,
               min_delta: float = 1e-6,
               save_path: str   = None,
               mode: str = 'min'):
    assert mode in ('min', 'max'), f"mode must be 'min' or 'max', got {mode!r}"
    self.patience   = patience
    self.min_delta  = min_delta
    self.save_path  = save_path
    self.mode       = mode
    self.best_score = float('inf') if mode == 'min' else -float('inf')
    self.counter    = 0
    self.stopped    = False

  @property
  def best_loss(self):
    return self.best_score

  def _is_improvement(self, score: float) -> bool:
    if self.mode == 'min':
      return score < self.best_score - self.min_delta
    return score > self.best_score + self.min_delta

  def __call__(self,
               score: float,
               model: nn.Module) -> bool:
    if self._is_improvement(score):
      self.best_score = score
      self.counter    = 0
      if self.save_path:
        torch.save(model.state_dict(), self.save_path)
    else:
      self.counter    += 1
      if self.counter >= self.patience:
        self.stopped  = True
    return self.stopped


def get_best_epoch(history: Dict, monitor: str) -> int:
  """Return the index of the epoch that EarlyStopping would consider "best"
  for the given cfg.AE_MONITOR, using the exact same min/max rule as
  EarlyStopping itself ('val_auroc' -> max, everything else -> min).

  This is the SINGLE shared definition of "best epoch" used by:
    - scripts/train.py  (to report best_val_loss / best_val_auroc etc. in
      final_results.json at the epoch that was actually selected, instead of
      the global min/max over all epochs which may belong to a different,
      non-selected epoch)
    - src/visual.py      (plot_training_history, to draw the "best epoch"
      marker on the training curves)

  Previously this same 3-line if/elif/else was duplicated inline inside
  src/visual.py while scripts/train.py used an unrelated (and misleading)
  `min(history['val_loss'])` over ALL epochs. Factoring it out here removes
  that duplication and guarantees both call-sites agree on what "best" means.
  """
  if monitor == 'val_auroc':
    return int(np.argmax(history['val_auroc']))
  elif monitor == 'val_loss_normal':
    return int(np.argmin(history['val_loss_normal']))
  else:
    return int(np.argmin(history['val_loss']))


def aggregate_score_torch(error_map: torch.Tensor, cfg) -> torch.Tensor:
  b = error_map.size(0)
  flat = error_map.reshape(b, -1)
  if cfg.SCORE_METHOD == 'mean':
    return flat.mean(dim=1)
  elif cfg.SCORE_METHOD == 'max':
    return flat.max(dim=1).values
  elif cfg.SCORE_METHOD == 'topk':
    k = max(1, int(flat.size(1) * cfg.SCORE_TOPK_PERCENT / 100.0))
    topk_vals, _ = flat.topk(k, dim=1)
    return topk_vals.mean(dim=1)
  else:
    raise ValueError(f'Unknown SCORE_METHOD: {cfg.SCORE_METHOD!r}')


def train_autoencoder(
    ae,
    extractor,
    normal_loader,
    val_loader,
    cfg
) -> Dict:
    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.AE_LR, weight_decay=cfg.AE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.AE_LR_STEP, gamma=cfg.AE_LR_GAMMA)
    criterion = get_criterion(cfg)

    save_path = os.path.join(cfg.SAVE_PATH, 'best_autoencoder.pth')
    monitor = cfg.AE_MONITOR
    assert monitor in ('val_loss', 'val_auroc', 'val_loss_normal'), \
        f"Unknown cfg.AE_MONITOR: {monitor!r}"
    es_mode = 'max' if monitor == 'val_auroc' else 'min'
    early_stop = EarlyStopping(patience=cfg.AE_PATIENCE, save_path=save_path, mode=es_mode)

    history = {'train_loss'      : [],
               'val_loss'        : [],
               'val_loss_normal' : [],
               'val_auroc'       : [],
               'lr'              : []}
    extractor.eval()
    for epoch in range(1, cfg.AE_EPOCHS + 1):
      # ── TRAIN ───────────────────────────────────────────────
      ae.train()
      train_loss = 0.0

      for norm_t, _, _, _, _, _ in normal_loader:
        norm_t = norm_t.to(cfg.DEVICE)
        with torch.no_grad():
          feats = extractor(norm_t)
          feats = extractor.normalize(feats)

        recon = ae(feats)
        loss = criterion(recon, feats)
        b_size = feats.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * b_size
      train_loss /= len(normal_loader.dataset)

      # ── VALIDATION ──────────────────────────────────────────
      ae.eval()
      val_loss = 0.0
      normal_loss_sum, normal_loss_n = 0.0, 0
      all_val_scores = []
      all_val_targets = []

      with torch.no_grad():
        for norm_t, _, _, _, batch_labels, _ in val_loader:
          norm_t = norm_t.to(cfg.DEVICE)
          feats  = extractor(norm_t)
          feats  = extractor.normalize(feats)
          recon = ae(feats)

          v_loss = criterion(recon, feats)
          val_loss += v_loss.item() * feats.size(0)
          error_map_raw   = elementwise_error_map(feats, recon, criterion)   # [B, H, W] native resolution
          per_sample_loss = error_map_raw.mean(dim=(1, 2))                    # [B] (diagnostic only, unaffected)

          # IMPORTANT (fix for the train/eval scoring-resolution mismatch):
          # The anomaly score used for val_auroc (which drives EarlyStopping /
          # checkpoint selection via cfg.AE_MONITOR) must be computed with the
          # EXACT SAME post-processing pipeline as the final reported metrics
          # in score_dataset_split(): upsample to cfg.IMAGE_SIZE, Gaussian-smooth
          # with cfg.HEATMAP_SIGMA, then aggregate with cfg.SCORE_METHOD.
          # We therefore route every sample through upsample_and_smooth() +
          # aggregate_score() here instead of aggregate_score_torch() on the
          # raw, native-resolution map. This is done on CPU (cv2/scipy have no
          # GPU batched equivalent) which is slightly slower per epoch, but
          # only affects the validation pass, not the training forward/backward.
          error_map_raw_np = error_map_raw.detach().cpu().numpy()
          batch_scores = np.empty(error_map_raw_np.shape[0], dtype=np.float32)
          for i in range(error_map_raw_np.shape[0]):
            smoothed_map = upsample_and_smooth(
                error_map_raw_np[i],
                sigma=cfg.HEATMAP_SIGMA,
                out_size=cfg.IMAGE_SIZE)
            batch_scores[i] = aggregate_score(smoothed_map, cfg)

          batch_y = np.array([1 if l == 'anomaly' else 0 for l in batch_labels])
          normal_mask = (batch_y == 0)
          if normal_mask.any():
            normal_loss_sum += per_sample_loss.cpu().numpy()[normal_mask].sum()
            normal_loss_n   += int(normal_mask.sum())

          all_val_scores.extend(batch_scores.tolist())
          all_val_targets.extend(batch_y.tolist())
      val_loss /= len(val_loader.dataset)
      val_loss_normal = normal_loss_sum / max(normal_loss_n, 1)

      all_val_targets = np.array(all_val_targets)
      all_val_scores = np.array(all_val_scores)

      if len(np.unique(all_val_targets)) > 1:
        val_auroc = roc_auc_score(all_val_targets, all_val_scores)
      else:
        val_auroc = 0.5

      scheduler.step()
      lr_now = optimizer.param_groups[0]['lr']

      history['train_loss']     .append(train_loss)
      history['val_loss']       .append(val_loss)
      history['val_loss_normal'].append(val_loss_normal)
      history['val_auroc']      .append(val_auroc)
      history['lr']             .append(lr_now)

      monitor_score = {'val_loss': val_loss,
                        'val_loss_normal': val_loss_normal,
                        'val_auroc': val_auroc}[monitor]

      print(f'Epoch [{epoch:4d}/{cfg.AE_EPOCHS}]  '
              f'TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  '
              f'ValLossNormal={val_loss_normal:.4f}  '
              f'ValAUROC={val_auroc:.4f}  '
              f'Monitor[{monitor}]={monitor_score:.4f}  '
              f'LR={lr_now:.2e}  (patience {early_stop.counter}/{early_stop.patience})')
      if early_stop(monitor_score, ae):
        print(f'\nEarly stopping at epoch {epoch}  '
              f'(best {monitor}={early_stop.best_score:.6f})')
        break
    # Load best weights
    ae.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    print(f'\n✓ Best autoencoder loaded from {save_path}  (monitor={monitor})')
    return history


def upsample_and_smooth(
    err_map  : np.ndarray,
    sigma    : float,
    out_size : Tuple[int, int]
) -> np.ndarray:
    """Resize a native-resolution error map up to out_size then Gaussian-smooth it.

    This is the SINGLE shared post-processing step between:
      (a) training-time validation scoring (train_autoencoder -> val_auroc,
          used by EarlyStopping to select the checkpoint), and
      (b) final scoring (score_dataset_split -> process_single_heatmap,
          used for the reported test/val/train AUC-ROC, F1, etc.)

    Both call-sites MUST route through this exact function (not a
    reimplementation) so that the criterion used to pick the "best" model
    is provably the same criterion used to report its final performance.
    Resizing (bilinear) and Gaussian blur are both non-linear w.r.t. the
    spatial ranking of error values, so computing them at only one of the
    two call-sites would let the two rankings diverge silently.
    """
    score_map_up = cv2.resize(
        err_map,
        (out_size[1], out_size[0]),
        interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(score_map_up, sigma=sigma)


def process_single_heatmap(
    feat_t       :  torch.Tensor,       # [C, H, W] tensor
    recon_t      :  torch.Tensor,       # [C, H, W] tensor
    sigma        :  float,
    out_size     :  Tuple[int,int],
    criterion    :  nn.Module = None
) -> Tuple[np.ndarray, np.ndarray]:
    if criterion is not None and isinstance(criterion, (SSIMLoss, CombinedLoss)):
      err_map = criterion.dissimilarity_map(
          recon_t.unsqueeze(0), feat_t.unsqueeze(0)
      ).squeeze(0).numpy()
    elif criterion is not None and isinstance(criterion, nn.L1Loss):
      err_map = (feat_t - recon_t).abs().mean(dim=0).numpy()
    else:
      # Default / nn.MSELoss case.
      err_map = ((feat_t - recon_t) ** 2).mean(dim=0).numpy()

    raw_map = upsample_and_smooth(err_map, sigma=sigma, out_size=out_size)

    s_min, s_max = raw_map.min(), raw_map.max()
    normalized_map = (raw_map - s_min) / ((s_max - s_min) + 1e-8)

    return raw_map, normalized_map


def aggregate_score(raw_map: np.ndarray, cfg) -> float:
  if cfg.SCORE_METHOD == 'mean':
    return float(raw_map.mean())
  elif cfg.SCORE_METHOD == 'max':
    return float(raw_map.max())
  elif cfg.SCORE_METHOD == 'topk':
    flat = raw_map.reshape(-1)
    k = max(1, int(flat.size * cfg.SCORE_TOPK_PERCENT / 100.0))
    topk_vals = np.partition(flat, -k)[-k:]
    return float(topk_vals.mean())
  else:
    raise ValueError(f'Unknown SCORE_METHOD: {cfg.SCORE_METHOD!r}')


def score_dataset_split(
    loader,
    extractor  : nn.Module,
    ae         : nn.Module,
    cfg,
    desc       : str = 'Scoring'
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

  extractor.eval()
  ae.eval()

  criterion = get_criterion(cfg)

  img_score, y_true, paths, labels, heatmaps, orig_imgs, preproc_imgs = [], [], [], [], [], [], []

  with torch.no_grad():
    for norm_t, orig_t, preproc_t, batch_paths, batch_labels, _ in tqdm(loader, desc=desc):

      inputs_device = norm_t.to(cfg.DEVICE)
      feats_device  = extractor(inputs_device)
      feats_device  = extractor.normalize(feats_device)
      recon_device  = ae(feats_device)
      feats_cpu = feats_device.cpu()
      recon_cpu = recon_device.cpu()
      for i in range(feats_cpu.size(0)):
        raw_map, normalized_map = process_single_heatmap(
            feats_cpu[i],
            recon_cpu[i],
            sigma = cfg.HEATMAP_SIGMA,
            out_size = cfg.IMAGE_SIZE,
            criterion = criterion
        )

        img_score.append(aggregate_score(raw_map, cfg))
        y_true.append(1 if batch_labels[i] == 'anomaly' else 0)
        paths.append(batch_paths[i])
        labels.append(batch_labels[i])
        heatmaps.append(normalized_map)
        orig_imgs.append(orig_t[i].permute(1, 2, 0).numpy())
        preproc_imgs.append(preproc_t[i].permute(1, 2, 0).numpy())

  return (np.array(img_score, dtype=np.float32),
        np.array(y_true, dtype=int),
        paths, labels, heatmaps, orig_imgs, preproc_imgs)