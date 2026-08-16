"""Training loop, EarlyStopping, และ scoring/heatmap pipeline

Training loop, EarlyStopping, and the scoring/heatmap pipeline.
"""
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.losses import get_criterion, elementwise_error_map
from src.optimes import get_optimizer


class EarlyStopping:
  """หยุดเทรนเมื่อ metric ที่ monitor ไม่ดีขึ้นติดต่อกันเกิน patience epoch
  พร้อมเซฟ checkpoint ที่ดีที่สุดไว้อัตโนมัติทุกครั้งที่ดีขึ้น

  Stops training once the monitored metric hasn't improved for
  `patience` consecutive epochs, automatically saving the best
  checkpoint every time it does improve.
  """
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
  """คืน index ของ epoch ที่ EarlyStopping จะถือว่า "ดีที่สุด" ตาม
  cfg.AE_MONITOR ที่ให้มา โดยใช้กฎ min/max เดียวกันเป๊ะกับ EarlyStopping เอง
  ('val_auroc' -> max, อย่างอื่น -> min)

  นี่คือนิยาม "best epoch" ที่ใช้ร่วมกันจุดเดียว (SINGLE shared definition)
  ระหว่าง:
    - scripts/train.py  (รายงาน best_val_loss / best_val_auroc ฯลฯ ใน
      final_results.json ที่ epoch ที่ถูกเลือกจริง แทนที่จะเป็น global
      min/max ข้ามทุก epoch ซึ่งอาจเป็นคนละ epoch กับที่ถูกเลือกจริง)
    - src/visual.py      (plot_training_history วาดจุด "best epoch"
      บนกราฟ training curve)

  เดิมโค้ด if/elif/else 3 บรรทัดนี้ถูก duplicate ไว้ใน src/visual.py ตรงๆ
  ในขณะที่ scripts/train.py ใช้ `min(history['val_loss'])` ข้ามทุก epoch
  ที่ไม่เกี่ยวข้องกัน (และทำให้เข้าใจผิด) การแยกออกมาเป็นฟังก์ชันนี้กำจัด
  ความซ้ำซ้อนนั้น และรับประกันว่าทั้งสองจุดเรียกเห็นตรงกันว่า "ดีที่สุด" คืออะไร

  Return the index of the epoch that EarlyStopping would consider "best"
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
  """รวม error map (batched, บน GPU/CPU tensor) เป็น score เดียวต่อภาพ
  ตาม cfg.SCORE_METHOD — ใช้ตอนอยากได้ความเร็วแบบ batched (ไม่ผ่าน
  upsample/blur เต็มรูปแบบ); ดู aggregate_score() ด้านล่างสำหรับเวอร์ชัน
  numpy ต่อภาพเดียวที่ pipeline การ scoring จริงใช้

  Aggregate a batched error map (torch tensor) into a single per-image
  score according to cfg.SCORE_METHOD — used when batched speed matters
  (bypassing the full upsample/blur path); see aggregate_score() below
  for the per-image numpy version the real scoring pipeline uses.
  """
  b = error_map.size(0)
  flat = error_map.reshape(b, -1)
  score_method = cfg.SCORE_METHOD.strip().lower()
  if score_method == 'mean':
    return flat.mean(dim=1)
  elif score_method == 'max':
    return flat.max(dim=1).values
  elif score_method == 'topk':
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
    """เทรน autoencoder บนภาพปกติ (good) เท่านั้น พร้อม validate ทุก epoch
    ด้วย pipeline scoring เดียวกับที่ใช้รายงานผลจริง แล้วคืน training history

    Train the autoencoder on normal (good) images only, validating every
    epoch through the exact same scoring pipeline used for final reported
    metrics, and return the training history.
    """
    optimizer = get_optimizer(cfg, ae.parameters())
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
      # ── TRAIN / เทรน ────────────────────────────────────────
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

      # ── VALIDATION / ตรวจสอบ ──────────────────────────────────
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
          error_map_raw   = elementwise_error_map(feats, recon, criterion)   # [B, H, W] ความละเอียดดิบ / native resolution
          per_sample_loss = error_map_raw.mean(dim=(1, 2))                    # [B] (สำหรับ diagnostic เท่านั้น ไม่กระทบผลอื่น / diagnostic only, unaffected)

          # สำคัญ (แก้ปัญหา train/eval scoring-resolution mismatch):
          # anomaly score ที่ใช้คำนวณ val_auroc (ตัวขับ EarlyStopping /
          # การเลือก checkpoint ผ่าน cfg.AE_MONITOR) ต้องคำนวณด้วย
          # post-processing pipeline ที่เหมือนกันเป๊ะกับ metric สุดท้ายที่
          # รายงานใน score_dataset_split(): upsample ไปที่ cfg.IMAGE_SIZE,
          # Gaussian-smooth ด้วย cfg.HEATMAP_SIGMA แล้วรวมด้วย
          # cfg.SCORE_METHOD เราจึงส่งทุก sample ผ่าน upsample_and_smooth()
          # + aggregate_score() ที่นี่ แทนที่จะใช้ aggregate_score_torch()
          # บน raw map ที่ความละเอียดดิบ ทำบน CPU (cv2/scipy ไม่มี batched
          # GPU version) ซึ่งช้ากว่าเล็กน้อยต่อ epoch แต่กระทบแค่ validation
          # pass เท่านั้น ไม่กระทบ training forward/backward
          #
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
          #
          # กรณี SCORE_METHOD='structcore' เป็นข้อยกเว้น: StructCore ต้อง fit
          # μ/σ/λ_auto จาก training set ทั้งชุดผ่าน AE ตัวปัจจุบันก่อนใช้งาน
          # ได้ (ดู fit_structcore_stats()) แต่ AE เปลี่ยน weight ทุก epoch
          # ระหว่างเทรน การ fit ใหม่ทุก epoch จะช้าเกินไปเพราะต้อง pass เต็ม
          # ผ่าน normal_loader ทั้งชุดในทุก epoch จึงใช้ max pooling ธรรมดา
          # เป็น proxy สำหรับ val_auroc ระหว่างเทรนแทน (เร็ว ไม่ต้อง fit)
          # ส่วนตัวเลขที่รายงานจริงตอนจบ (score_dataset_split) จะ fit และ
          # ใช้ StructCore เต็มรูปแบบ — เป็นข้อจำกัดทางสถาปัตยกรรมที่หลีก
          # เลี่ยงไม่ได้ ไม่ใช่บั๊ก ต้องระบุไว้ใน thesis ถ้าใช้ SCORE_METHOD
          # นี้ว่า metric ที่ใช้เลือก checkpoint กับที่รายงานผลไม่ใช่ตัว
          # เดียวกันเป๊ะในกรณีนี้เท่านั้น
          #
          # SCORE_METHOD='structcore' is an exception: StructCore must fit
          # μ/σ/λ_auto from the entire training set through the current AE
          # before it can be used (see fit_structcore_stats()), but the AE's
          # weights change every epoch during training. Refitting every
          # epoch would be too slow (a full pass over normal_loader every
          # epoch), so plain max pooling is used as a proxy for val_auroc
          # during training instead (fast, no fitting needed). The final
          # reported numbers (score_dataset_split) fit and use full
          # StructCore. This is an unavoidable architectural limitation, not
          # a bug — if this SCORE_METHOD is used, note in the thesis that
          # the checkpoint-selection metric and the reported metric are not
          # exactly the same in this one case.
          error_map_raw_np = error_map_raw.detach().cpu().numpy()
          batch_scores = np.empty(error_map_raw_np.shape[0], dtype=np.float32)
          for i in range(error_map_raw_np.shape[0]):
            smoothed_map = upsample_and_smooth(
                error_map_raw_np[i],
                sigma=cfg.HEATMAP_SIGMA,
                out_size=cfg.IMAGE_SIZE)
            batch_scores[i] = (
                float(smoothed_map.max()) if cfg.SCORE_METHOD.strip().lower() == 'structcore'
                else aggregate_score(smoothed_map, cfg))

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
    # โหลด weight ที่ดีที่สุดกลับมา / Load best weights
    ae.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    print(f'\n✓ Best autoencoder loaded from {save_path}  (monitor={monitor})')
    return history


def upsample_and_smooth(
    err_map  : np.ndarray,
    sigma    : float,
    out_size : Tuple[int, int]
) -> np.ndarray:
    """ขยาย error map ความละเอียดดิบขึ้นไปที่ out_size แล้ว Gaussian-smooth

    นี่คือขั้นตอน post-processing ที่ใช้ร่วมกันจุดเดียว (SINGLE shared step)
    ระหว่าง:
      (a) การ scoring ตอน validation ระหว่างเทรน (train_autoencoder ->
          val_auroc ที่ EarlyStopping ใช้เลือก checkpoint) และ
      (b) การ scoring ครั้งสุดท้าย (score_dataset_split ->
          process_single_heatmap ที่ใช้รายงาน test/val/train AUC-ROC, F1 ฯลฯ)

    ทั้งสองจุดเรียก**ต้อง**ผ่านฟังก์ชันนี้เป๊ะๆ (ไม่ implement ซ้ำ) เพื่อให้
    criterion ที่ใช้เลือกโมเดล "ดีที่สุด" เป็นตัวเดียวกันเป๊ะกับที่ใช้รายงาน
    ผลลัพธ์สุดท้าย การ resize (bilinear) และ Gaussian blur ทั้งคู่เป็น
    non-linear เทียบกับลำดับ (ranking) เชิงพื้นที่ของค่า error ถ้าคำนวณแค่
    จุดใดจุดหนึ่งจากสองจุดนี้ จะทำให้ ranking ทั้งสองเบี่ยงเบนออกจากกันเงียบๆ
    โดยไม่มีการแจ้งเตือน

    Resize a native-resolution error map up to out_size then Gaussian-smooth it.

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
    """คำนวณ heatmap ของภาพเดียว (raw + normalized [0,1]) จาก feature กับ
    reconstruction ของภาพนั้น

    Compute a single image's heatmap (raw + normalized to [0,1]) from its
    feature map and reconstruction.
    """
    if criterion is not None:
      # เรียกผ่าน elementwise_error_map() ตัวเดียวกับที่ train_autoencoder()
      # ใช้คำนวณ val_auroc ตอน validation (ตัวสัญญาณที่ EarlyStopping/การ
      # เลือก checkpoint พึ่งพา)
      #
      # เดิมจุดนี้เคยมี isinstance chain ของตัวเองแยกต่างหาก ทำให้การเพิ่ม
      # loss ใหม่ (เช่น CosineLoss / CosineMSELoss) เข้า
      # elementwise_error_map() เพียงจุดเดียว **ไม่ propagate** มาที่ฟังก์ชัน
      # นี้ — score_dataset_split() จะตกไปที่ branch plain-MSE ด้านล่าง
      # เงียบๆ ทั้งที่โมเดลถูกเทรนและเลือก checkpoint ด้วยคนละ criterion
      # นี่คือ failure mode เดียวกับที่ docstring ของ elementwise_error_map()
      # เตือนไว้ตรงๆ เพียงแต่เกิดซ้ำผ่าน dispatch logic คนละชุด การเรียกผ่าน
      # ฟังก์ชันเดียวกันตรงนี้กำจัด copy ที่สองทิ้ง ทำให้สองจุดเรียกเบี่ยงเบน
      # จากกันไม่ได้อีก (เคยเป็นบั๊กจริงที่พบและแก้ไปแล้ว)
      #
      # Route through the SAME elementwise_error_map() that
      # train_autoencoder()'s validation pass uses to compute val_auroc
      # (the signal EarlyStopping/checkpoint selection is based on).
      #
      # This used to be a second, independently-maintained isinstance
      # chain here. That meant adding a new loss (e.g. CosineLoss /
      # CosineMSELoss) to elementwise_error_map() alone would NOT
      # propagate to this function -- score_dataset_split() would then
      # silently fall through to the plain-MSE branch below while the
      # model was actually trained and checkpoint-selected on a
      # different criterion. That's the exact failure mode
      # elementwise_error_map()'s own docstring warns about, just
      # reintroduced via a second copy of the dispatch logic. Delegating
      # here removes the second copy so both call-sites can never diverge.
      err_map = elementwise_error_map(
          feat_t.unsqueeze(0), recon_t.unsqueeze(0), criterion
      ).squeeze(0).numpy()
    else:
      # ไม่ได้ส่ง criterion มา (เก็บไว้เพื่อ backward compatibility) -> ใช้ MSE ธรรมดา
      # No criterion supplied (kept for backward compatibility) -> plain MSE.
      err_map = ((feat_t - recon_t) ** 2).mean(dim=0).numpy()

    raw_map = upsample_and_smooth(err_map, sigma=sigma, out_size=out_size)

    s_min, s_max = raw_map.min(), raw_map.max()
    normalized_map = (raw_map - s_min) / ((s_max - s_min) + 1e-8)

    return raw_map, normalized_map


def compute_phi(raw_map: np.ndarray, topk_ratio: float) -> np.ndarray:
  """คำนวณ structural descriptor φ(S) = [σ_S, topk_mean_r, TV(S)] ตาม
  StructCore (Chae et al. 2026, arXiv:2602.17048) — สรุป error map เป็น
  vector 3 มิติที่จับ 3 คุณสมบัติเสริมกัน: (1) การกระจายตัวโดยรวม
  (σ_S = ส่วนเบี่ยงเบนมาตรฐานของทั้ง map), (2) ความเข้มข้นของ tail
  (topk_mean_r = ค่าเฉลี่ยของ pixel ที่ค่าสูงสุด top-`topk_ratio`),
  (3) ความขรุขระเชิงพื้นที่ (TV = total variation — ผลรวมค่าสัมบูรณ์ของ
  ผลต่างระหว่างพิกเซลข้างเคียง หารด้วยจำนวนพิกเซลทั้งหมด)

  ต่างจาก max pooling เดี่ยวๆ (ที่ดูแค่ pixel เดียว) φ(S) จับ "รูปร่าง"
  ของความผิดปกติทั้ง map ไว้ด้วย — defect ที่กระจายเป็นหย่อมเล็กๆ
  หลายจุด หรือ defect ที่ diffuse (ไม่มี peak เดียวที่ชัดเจน) จะสะท้อน
  ออกมาใน σ_S/TV แม้ max ของ map จะไม่ได้สูงผิดปกติมากก็ตาม

  Compute the structural descriptor φ(S) = [σ_S, topk_mean_r, TV(S)]
  per StructCore (Chae et al. 2026, arXiv:2602.17048) — summarizes the
  error map into a 3D vector capturing three complementary properties:
  (1) overall dispersion (σ_S = standard deviation of the whole map),
  (2) tail concentration (topk_mean_r = mean of the top-`topk_ratio`
  highest-value pixels), (3) spatial roughness (TV = total variation —
  sum of absolute differences between neighboring pixels, divided by
  the total pixel count).

  Unlike max pooling alone (which only looks at a single pixel), φ(S)
  also captures the "shape" of the anomaly across the whole map —
  defects that are spread across several small patches, or diffuse
  defects with no single sharp peak, show up in σ_S/TV even when the
  map's raw maximum isn't unusually high.
  """
  # แปลงเป็น 1 มิติก่อน เพื่อคำนวณสถิติของทั้ง map รวมกัน (ไม่แยก
  # spatial structure ในสองบรรทัดแรกนี้ — TV ด้านล่างต่างหากที่ใช้ shape 2D)
  # Flatten to 1D first, to compute statistics over the whole map (the
  # first two lines below don't care about spatial structure — TV below
  # is the only part that uses the 2D shape).
  flat = raw_map.reshape(-1)

  # องค์ประกอบที่ 1: σ_S — ส่วนเบี่ยงเบนมาตรฐานของทั้ง map (การกระจายตัวโดยรวม)
  # Component 1: σ_S — standard deviation of the whole map (overall dispersion)
  sigma_s = float(flat.std())

  # องค์ประกอบที่ 2: topk_mean_r — เฉลี่ยของ pixel ที่ค่าสูงสุด top-topk_ratio
  # (เช่น topk_ratio=0.01 คือ top 1% ของพิกเซลทั้งหมด)
  # k คือจำนวนพิกเซลจริง ปัดขึ้นอย่างน้อย 1 พิกเซลเสมอ กัน map เล็กมากจน k=0
  #
  # Component 2: topk_mean_r — mean of the top-topk_ratio highest-value
  # pixels (e.g. topk_ratio=0.01 = top 1% of all pixels).
  # k is the actual pixel count, floored at 1 so a very small map never
  # yields k=0.
  k = max(1, int(round(flat.size * topk_ratio)))
  topk_vals = np.partition(flat, -k)[-k:]   # k ค่าที่สูงที่สุด (ไม่เรียงลำดับ) / the k highest values (unordered)
  topk_mean = float(topk_vals.mean())

  # องค์ประกอบที่ 3: TV(S) — total variation คือผลรวมค่าสัมบูรณ์ของผลต่าง
  # ระหว่างพิกเซลข้างเคียงตามแนวแกน 0 (แถว) และแกน 1 (คอลัมน์) หารด้วย
  # จำนวนพิกเซลทั้งหมด (H*W) — วัดความ "ขรุขระ" เชิงพื้นที่ของ map: map
  # ที่ error กระจายเป็นหย่อมๆ ไม่เรียบจะมี TV สูงกว่า map ที่ error
  # เปลี่ยนแปลงราบเรียบ แม้ max/mean จะเท่ากันก็ตาม
  #
  # Component 3: TV(S) — total variation: sum of absolute differences
  # between neighboring pixels along axis 0 (rows) and axis 1 (columns),
  # divided by the total pixel count (H*W) — measures the map's spatial
  # "roughness": a map where error is scattered in uneven patches has
  # higher TV than one where error changes smoothly, even if their
  # max/mean are identical.
  tv = float(
      np.abs(np.diff(raw_map, axis=0)).sum() +   # |S[i+1,j] - S[i,j]| รวมทุกคู่ / summed over every row-pair
      np.abs(np.diff(raw_map, axis=1)).sum()     # |S[i,j+1] - S[i,j]| รวมทุกคู่ / summed over every column-pair
  ) / raw_map.size

  # รวม 3 องค์ประกอบเป็น φ(S) ตามลำดับที่ paper กำหนด (สำคัญ: ลำดับนี้
  # ต้องตรงกับลำดับที่ fit_structcore_stats() ใช้ fit mu/sigma ด้วย ไม่งั้น
  # D_struct ใน aggregate_score() จะคำนวณผิดมิติ)
  #
  # Combine the 3 components into φ(S) in the order the paper defines
  # (important: this order must match the order fit_structcore_stats()
  # fits mu/sigma in, or D_struct in aggregate_score() would compute
  # against mismatched dimensions).
  return np.array([sigma_s, topk_mean, tv], dtype=np.float64)


def fit_structcore_stats(normal_loader, extractor, ae, criterion, cfg) -> dict:
  """Fit สถิติ StructCore (μ, σ ของ φ(S), และ λ_auto) จาก **training set
  (normal เท่านั้น)** — เรียกครั้งเดียวหลังเทรนเสร็จ ก่อนเริ่ม scoring
  val/test จริง (ไม่ใช่ระหว่างเทรน เพราะ AE เปลี่ยน weight ทุก epoch การ
  fit ใหม่ทุก epoch จะช้าเกินไป — ดูรายละเอียดที่ elif
  cfg.SCORE_METHOD=='structcore' ใน train_autoencoder() ด้านล่างที่ใช้
  'max' pooling เป็น proxy ระหว่างเทรนแทน)

  Fit ด้วย train-good เท่านั้น หมายความว่าขั้นตอนนี้**ไม่ใช้ anomaly
  label เลย** (label-free 100%) ต่างจาก percentile threshold ที่ต้องรู้
  label 'normal' ของ validation set

  μ, σ: ค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานต่อมิติของ φ(S) บน training set
  λ_auto: น้ำหนักที่ปรับสเกลของ D_struct ให้ใกล้เคียงกับสเกลของ S_base
  (max pooling) โดยอัตโนมัติ = std(S_base บน train) / (std(D_struct บน
  train) + eps) — เพื่อไม่ต้อง tune λ เอง (ตามที่ paper แนะนำ)

  Fit StructCore statistics (φ(S)'s μ, σ, and λ_auto) from the
  **training set (normal only)** — called once after training finishes,
  before scoring val/test (not during training, since the AE's weights
  change every epoch and refitting every epoch would be too slow — see
  the cfg.SCORE_METHOD=='structcore' branch inside train_autoencoder()
  below, which uses 'max' pooling as a proxy during training instead).

  Fitting on train-good only means this step uses NO anomaly labels at
  all (100% label-free), unlike the percentile threshold which needs to
  know the validation set's 'normal' labels.

  μ, σ: per-dimension mean/std of φ(S) over the training set.
  λ_auto: automatic scale-matching weight between D_struct and S_base
  (max pooling) = std(S_base on train) / (std(D_struct on train) + eps)
  — avoids manually tuning λ (as recommended by the paper).
  """
  # ต้องเป็น eval mode ทั้งคู่ — นี่คือ inference pass ล้วนๆ ไม่มีการเทรน
  # ต่อ (ไม่ต้องมี dropout/batchnorm แบบ train mode)
  # Both must be in eval mode — this is a pure inference pass, no further
  # training (no dropout/batchnorm train-mode behavior needed).
  extractor.eval()
  ae.eval()

  phis, base_scores = [], []   # เก็บ φ(S) และ max-pooling score ของทุกภาพใน training set / collect φ(S) and the max-pooling score for every training image
  with torch.no_grad():
    for norm_t, _, _, _, _, _ in normal_loader:   # label (ตัวที่ 5) ถูกทิ้งตรงๆ — ไม่จำเป็นเพราะ normal_loader มีแต่ normal อยู่แล้ว / label (5th item) is discarded outright — unneeded since normal_loader is already normal-only
      norm_t = norm_t.to(cfg.DEVICE)
      feats = extractor(norm_t)
      feats = extractor.normalize(feats)
      recon = ae(feats)
      # error map ดิบต่อภาพ ด้วย criterion เดียวกับที่ AE ถูกเทรนจริง
      # (LOSS ที่ตั้งใน cfg) — สอดคล้องกับ error map ที่ AE "เห็น" ตอนเทรน
      # Raw per-image error map, using the same criterion the AE was
      # actually trained on (the LOSS set in cfg) — consistent with the
      # error map the AE "saw" during training.
      err_map_raw = elementwise_error_map(feats, recon, criterion)  # [B, H, W]
      err_map_raw_np = err_map_raw.detach().cpu().numpy()
      for i in range(err_map_raw_np.shape[0]):
        # upsample + smooth ให้เหมือนกับ pipeline ตอน scoring จริงเป๊ะ
        # (score_dataset_split -> process_single_heatmap) ไม่งั้น φ(S) ที่
        # fit ไว้ตรงนี้จะไม่ตรงกับ φ(S) ที่คำนวณตอน inference จริง
        #
        # Upsample + smooth using the exact same pipeline as real scoring
        # (score_dataset_split -> process_single_heatmap), or the φ(S)
        # fit here wouldn't match the φ(S) computed at real inference time.
        smoothed = upsample_and_smooth(
            err_map_raw_np[i], sigma=cfg.HEATMAP_SIGMA, out_size=cfg.IMAGE_SIZE)
        phis.append(compute_phi(smoothed, cfg.STRUCTCORE_TOPK_RATIO))
        base_scores.append(float(smoothed.max()))   # S_base ของภาพนี้ (max pooling ธรรมดา ตามที่ paper กำหนดให้เป็น base) / this image's S_base (plain max pooling, the paper's chosen base score)

  phis = np.stack(phis)          # [N, 3] — N = จำนวนภาพทั้งหมดใน training set / N = total number of training images
  base_scores = np.array(base_scores)  # [N]

  # μ, σ ต่อมิติของ φ(S) บน training set — นิยาม "ปกติ" ในเชิงสถิติของ
  # structural descriptor (ทำหน้าที่คล้าย extractor.feat_mean/feat_std
  # ใน backbone_baseline.py แต่ fit บน φ(S) แทน feature map ดิบ)
  #
  # Per-dimension μ, σ of φ(S) over the training set — the statistical
  # definition of "normal" for the structural descriptor (plays a role
  # similar to extractor.feat_mean/feat_std in backbone_baseline.py, but
  # fit on φ(S) instead of the raw feature map).
  mu = phis.mean(axis=0)
  sigma = phis.std(axis=0)

  # D_struct ของทุกภาพใน training set เอง (diagonal Mahalanobis distance
  # ระหว่าง φ(S) ของแต่ละภาพกับ μ, σ ที่เพิ่ง fit ไป) — ใช้แค่เพื่อหา
  # scale ตามธรรมชาติของ D_struct เทียบกับ S_base เท่านั้น ไม่ได้เก็บค่า
  # นี้ไว้ใช้ที่อื่นต่อ
  #
  # D_struct for every training image itself (diagonal Mahalanobis
  # distance between each image's φ(S) and the μ, σ just fit) — used only
  # to discover D_struct's natural scale relative to S_base; this value
  # isn't kept for anything else afterward.
  d_struct_train = np.linalg.norm((phis - mu) / (sigma + cfg.STRUCTCORE_EPS), axis=1)

  # λ_auto จับคู่ scale ของ D_struct ให้ใกล้เคียงกับ scale ของ S_base
  # โดยอัตโนมัติ (เทียบ std ของสองฝั่งบน training set) — ป้องกันไม่ให้
  # เทอมใดเทอมหนึ่งใน S_hyb = S_base + λ*D_struct ครอบงำอีกเทอมเพราะหน่วย
  # ไม่ตรงกัน โดยไม่ต้อง grid-search หา λ เอง
  #
  # λ_auto automatically matches D_struct's scale to S_base's (comparing
  # their standard deviations over the training set) — prevents either
  # term in S_hyb = S_base + λ*D_struct from dominating the other purely
  # due to mismatched units, without having to grid-search for λ by hand.
  lambda_auto = float(
      base_scores.std() / (d_struct_train.std() + cfg.STRUCTCORE_EPS))

  # เก็บเป็น list ธรรมดา (ไม่ใช่ np.array) เพื่อให้ serialize เป็น JSON
  # ได้ตรงๆ ถ้าต้องการเซฟไว้ (io_utils.py pattern เดียวกับไฟล์อื่นในระบบนี้)
  # Store as plain lists (not np.array) so this is directly JSON-
  # serializable if saved to disk (same io_utils.py pattern used
  # elsewhere in this codebase).
  return {'mu': mu.tolist(), 'sigma': sigma.tolist(), 'lambda_auto': lambda_auto}


def aggregate_score(raw_map: np.ndarray, cfg, structcore_stats: dict = None) -> float:
  """รวม heatmap ของภาพเดียว (numpy array) เป็น anomaly score เดียว
  ตาม cfg.SCORE_METHOD ('mean' | 'max' | 'topk' | 'structcore')

  Aggregate a single image's heatmap (numpy array) into one anomaly
  score, according to cfg.SCORE_METHOD ('mean' | 'max' | 'topk' |
  'structcore').

  `structcore_stats`: ต้องส่งมาเมื่อ SCORE_METHOD='structcore' เท่านั้น
  (ผลลัพธ์จาก fit_structcore_stats() — เรียก error ถ้าไม่ส่งมา แทนที่จะ
  fallback ไปทำอะไรแบบเงียบๆ). `structcore_stats` must be supplied only
  when SCORE_METHOD='structcore' (the output of fit_structcore_stats())
  — raises instead of silently falling back to something else if missing.
  """
  score_method = cfg.SCORE_METHOD.strip().lower()
  if score_method == 'mean':
    return float(raw_map.mean())
  elif score_method == 'max':
    return float(raw_map.max())
  elif score_method == 'topk':
    flat = raw_map.reshape(-1)
    k = max(1, int(flat.size * cfg.SCORE_TOPK_PERCENT / 100.0))
    topk_vals = np.partition(flat, -k)[-k:]
    return float(topk_vals.mean())
  elif score_method == 'structcore':
    if structcore_stats is None:
      raise ValueError(
          "SCORE_METHOD='structcore' requires structcore_stats (call "
          "fit_structcore_stats() once after training finishes, then pass "
          "its result through score_dataset_split()'s structcore_stats "
          "argument).")
    # ดึง μ, σ, λ_auto ที่ fit ไว้แล้วจาก training set กลับมา (ผลลัพธ์จาก
    # fit_structcore_stats() — ดึง . tolist() กลับเป็น np.array เพื่อคำนวณ)
    # Retrieve the already-fit μ, σ, λ_auto from the training set (the
    # output of fit_structcore_stats() — converted back from .tolist()
    # into np.array for computation).
    mu = np.array(structcore_stats['mu'])
    sigma = np.array(structcore_stats['sigma'])
    lam = structcore_stats['lambda_auto']

    # φ(S) ของภาพนี้ ด้วย topk_ratio เดียวกับตอน fit (ต้องตรงกันเป๊ะ ไม่งั้น
    # เทียบกับ μ/σ ผิดสเกล)
    # φ(S) for this image, using the same topk_ratio as at fit time (must
    # match exactly, or comparing against μ/σ would be on the wrong scale).
    phi = compute_phi(raw_map, cfg.STRUCTCORE_TOPK_RATIO)

    # D_struct = diagonal Mahalanobis distance ของ φ(S) ภาพนี้ เทียบกับ
    # การกระจายตัวของ φ(S) บน training set — ยิ่งภาพนี้ "รูปร่าง" ของ error
    # ต่างจาก training set ปกติมากเท่าไหร่ ค่านี้ยิ่งสูง
    # D_struct = diagonal Mahalanobis distance between this image's φ(S)
    # and the training set's φ(S) distribution — the more this image's
    # error "shape" deviates from the normal training distribution, the
    # higher this value.
    d_struct = float(np.linalg.norm((phi - mu) / (sigma + cfg.STRUCTCORE_EPS)))

    # S_base = max pooling ธรรมดา (ตามที่ paper กำหนดเป็น base score)
    # S_base = plain max pooling (the paper's chosen base score).
    s_base = float(raw_map.max())

    # S_hyb = S_base + λ_auto * D_struct — รวม "ความรุนแรงที่จุดเดียว"
    # (S_base) เข้ากับ "ความผิดปกติของรูปร่างทั้ง map" (D_struct) โดย
    # λ_auto ปรับสเกลให้สองเทอมนี้มีน้ำหนักใกล้เคียงกันแล้ว (คำนวณไว้ตอน fit)
    # S_hyb = S_base + λ_auto * D_struct — combines "single-point severity"
    # (S_base) with "whole-map shape abnormality" (D_struct), with
    # λ_auto already scale-matching the two terms (computed at fit time).
    return s_base + lam * d_struct
  else:
    raise ValueError(f'Unknown SCORE_METHOD: {cfg.SCORE_METHOD!r}')


def score_dataset_split(
    loader,
    extractor  : nn.Module,
    ae         : nn.Module,
    cfg,
    desc       : str = 'Scoring',
    structcore_stats: dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
  """รัน inference บนทั้ง split (val/test) แล้วคืน score, label, heatmap,
  และภาพต้นฉบับของทุกภาพในนั้น — นี่คือฟังก์ชันที่ผลิตตัวเลข AUROC/F1/ฯลฯ
  ที่รายงานเป็นผลสุดท้าย

  `structcore_stats`: ต้องส่งมาถ้า cfg.SCORE_METHOD='structcore' (ผลลัพธ์
  จาก fit_structcore_stats() ที่ fit ครั้งเดียวหลังเทรนเสร็จ) ไม่ใช้กับ
  SCORE_METHOD อื่น

  Run inference over an entire split (val/test) and return the score,
  label, heatmap, and original image for every image in it — this is the
  function that produces the final reported AUROC/F1/etc.

  `structcore_stats`: required only if cfg.SCORE_METHOD='structcore' (the
  output of fit_structcore_stats(), fit once after training finishes);
  unused for any other SCORE_METHOD.
  """
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

        img_score.append(aggregate_score(raw_map, cfg, structcore_stats=structcore_stats))
        y_true.append(1 if batch_labels[i] == 'anomaly' else 0)
        paths.append(batch_paths[i])
        labels.append(batch_labels[i])
        heatmaps.append(normalized_map)
        orig_imgs.append(orig_t[i].permute(1, 2, 0).numpy())
        preproc_imgs.append(preproc_t[i].permute(1, 2, 0).numpy())

  return (np.array(img_score, dtype=np.float32),
        np.array(y_true, dtype=int),
        paths, labels, heatmaps, orig_imgs, preproc_imgs)