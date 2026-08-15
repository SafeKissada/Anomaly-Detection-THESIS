"""Loss functions สำหรับ feature-space autoencoder (ConvNeXt Stage2+Stage3,
z-score normalized) — ทุกตัวถูกเรียกผ่าน get_criterion(cfg) จุดเดียว

Loss functions for the feature-space autoencoder (z-score-normalized
ConvNeXt Stage2+Stage3 activations). Every loss below is constructed
through the single get_criterion(cfg) factory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
  """Cosine distance ต่อตำแหน่งพิกเซล คำนวณข้ามแกน channel

  Per-pixel cosine distance across the channel axis.

  Feature map ที่ reconstruct มี shape (B, C, H, W) — ที่แต่ละตำแหน่งเชิงพื้นที่
  (h, w) เวกเตอร์ยาว C ถูกมองเป็น "ทิศทาง" หนึ่งทิศทาง เพราะ scale-invariant
  โดยธรรมชาติ จึงไม่ต้องมีการ calibrate data_range/luminance และค่า loss
  จะ stationary ตลอดการเทรนไม่ว่าค่า feature ที่ normalize แล้วจะเลื่อนไปแค่ไหน
  (ต่างจาก SSIM แบบ pixel-space ที่ถูกดัดแปลงมาใช้กับ feature space ซึ่งจะไม่
  stationary แบบนี้ — ดูเหตุผลเต็มใน loss_functions_summary.md ว่าทำไม SSIM
  ถูกถอดออกจากโค้ดนี้)

  Reconstructed feature maps have shape (B, C, H, W); at each spatial
  position (h, w) the length-C vector is treated as a single direction.
  Scale-invariant by construction, so it needs no data_range/luminance
  calibration and stays stationary across training regardless of how the
  z-scored feature values drift (unlike a pixel-space SSIM adapted to
  feature space — see loss_functions_summary.md for why SSIM was removed
  from this codebase).

  ข้อควรระวัง / Caveat: scale-invariant โดยธรรมชาติ ทำให้เดี่ยวๆ แล้วมันไม่เคย
  ลงโทษ decoder ที่ output norm เพี้ยนไปจาก input เลย ลงโทษแค่ทิศทางเท่านั้น
  ถ้าไม่ได้ตั้งใจแบบนี้จริงๆ แนะนำใช้ CosineMSELoss แทน.
  Alone, it never penalizes the decoder for drifting the output norm away
  from the input's — only the direction. Prefer CosineMSELoss unless
  that's specifically what you want.
  """

  def __init__(self, eps: float = 1e-8):
    super().__init__()
    self.eps = eps  # stability epsilon ของ cosine_similarity / stability epsilon for cosine_similarity

  def _cosine_distance_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # cos_sim อยู่ในช่วง [-1, 1] ต่อตำแหน่ง (B, H, W) แล้วแปลงเป็น distance [0, 2]
    # cos_sim is in [-1, 1] per position (B, H, W); convert to distance in [0, 2]
    cos_sim = F.cosine_similarity(recon, target, dim=1, eps=self.eps)
    return 1.0 - cos_sim

  def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return self._cosine_distance_map(recon, target).mean()

  def dissimilarity_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """คืน error map ต่อตำแหน่งแบบไม่ reduce (สำหรับทำ heatmap)
    Returns the per-position error map, unreduced (used to build heatmaps)."""
    return self._cosine_distance_map(recon, target)


class CosineMSELoss(nn.Module):
  """ผสม cosine distance (ทิศทาง / cross-channel activation pattern) กับ MSE
  (ขนาด) — เทอม MSE คือสิ่งที่กัน scale-invariance ของ CosineLoss ไม่ให้ปล่อยให้
  output norm ของ decoder ลอยตัวโดยไม่ถูกลงโทษ

  Combines cosine distance (direction / cross-channel activation pattern)
  with MSE (magnitude) — the MSE term is what keeps CosineLoss's scale
  invariance from letting the decoder's output norm drift unpenalized.

  lam=1.0 = pure cosine, lam=0.0 = pure MSE.
  """

  def __init__(self, lam: float = 0.5, eps: float = 1e-8):
    super().__init__()
    assert 0.0 <= lam <= 1.0, "lam must be in [0, 1]"
    self.lam = lam
    self.eps = eps
    self.cosine = CosineLoss(eps=eps)
    self.mse = nn.MSELoss()

  def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_term = self.cosine._cosine_distance_map(recon, target).mean()
    mse_term = self.mse(recon, target)
    return self.lam * cos_term + (1.0 - self.lam) * mse_term

  def dissimilarity_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """คืน error map ต่อตำแหน่งแบบไม่ reduce (สำหรับทำ heatmap)
    Returns the per-position error map, unreduced (used to build heatmaps)."""
    cos_map = self.cosine._cosine_distance_map(recon, target)  # (B, H, W)
    mse_map = ((target - recon) ** 2).mean(dim=1)              # (B, H, W)
    return self.lam * cos_map + (1.0 - self.lam) * mse_map


def _huber_elementwise(diff: torch.Tensor, delta: float) -> torch.Tensor:
  """Huber loss ต่อ element (ก่อน reduce ใดๆ) สูตรตรงกับ nn.HuberLoss เป๊ะ:
      0.5 * diff**2                  ถ้า |diff| <  delta   (quadratic เหมือน MSE)
      delta * (|diff| - 0.5*delta)   ถ้า |diff| >= delta   (linear เหมือน MAE)
  แยกออกมาเป็นฟังก์ชันนี้ เพื่อให้ elementwise_error_map() (ด้านล่าง) และ
  process_single_heatmap() (src/engine.py) คำนวณสูตรเดียวกันเป๊ะ ไม่ต้อง
  maintain สูตรซ้ำสองที่ (ต้นเหตุของบั๊กที่เคยเจอใน engine.py มาก่อน)

  Per-element Huber loss (before any reduction), matching nn.HuberLoss's
  own formula exactly:
      0.5 * diff**2                  if |diff| <  delta   (quadratic, like MSE)
      delta * (|diff| - 0.5*delta)   if |diff| >= delta   (linear, like MAE)
  Factored out here so both elementwise_error_map() (below) and
  process_single_heatmap() (src/engine.py) compute it identically instead
  of maintaining two copies of the same formula (the root cause of a bug
  that was previously found and fixed in engine.py).
  """
  abs_diff = diff.abs()
  quadratic = 0.5 * diff ** 2
  linear = delta * (abs_diff - 0.5 * delta)
  return torch.where(abs_diff < delta, quadratic, linear)


def get_criterion(cfg) -> nn.Module:
  """สร้าง loss module จาก cfg.LOSS — จุดเดียวที่ map ชื่อ LOSS ไปเป็น class จริง
  Build the loss module from cfg.LOSS — the single place that maps a LOSS
  name string to its actual class.
  """
  loss_name = cfg.LOSS.upper()
  if loss_name == 'MSE':
    return nn.MSELoss()
  elif loss_name in ('MAE', 'L1'):
    return nn.L1Loss()
  elif loss_name in ('HUBER', 'SMOOTH_L1', 'SMOOTHL1'):
    return nn.HuberLoss(delta=cfg.HUBER_DELTA)
  elif loss_name == 'COS':
    return CosineLoss(eps=cfg.COS_EPS)
  elif loss_name in ('COS_MSE', 'COS+MSE', 'COSMSE'):
    return CosineMSELoss(lam=cfg.COS_LAM, eps=cfg.COS_EPS)
  else:
    raise ValueError(f"Unknown cfg.LOSS: {cfg.LOSS!r} (expected 'MSE', 'MAE', "
                     f"'HUBER', 'COS', or 'COS_MSE')")


def elementwise_error_map(feats: torch.Tensor, recon: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
  """Error map ต่อพิกเซล ใช้สำหรับ anomaly scoring — reduction ต้องตรงกับ
  `criterion` ที่โมเดลถูกเทรนจริงเป๊ะ ไม่งั้น score/AUROC/threshold ที่รายงาน
  จะถูกคำนวณจาก error metric คนละตัวกับที่ใช้เทรนโมเดลโดยไม่มีการแจ้งเตือนใดๆ
  (จุดนี้เคยมีบั๊กจริง — src/engine.py:process_single_heatmap() เคยมี dispatch
  logic คู่ขนานที่ไม่ sync กับฟังก์ชันนี้ ตอนนี้แก้แล้วให้เรียกฟังก์ชันนี้ตัวเดียว)

  Per-pixel error map used for anomaly scoring — MUST match the
  reduction the given `criterion` actually trains on, or the score/AUROC/
  threshold reported would silently be computed from a different error
  metric than the one the model was optimized for. (This was a real bug:
  src/engine.py:process_single_heatmap() used to have a parallel dispatch
  table that could drift out of sync with this one — it now delegates
  here instead.)
  """
  if isinstance(criterion, (CosineLoss, CosineMSELoss)):
    return criterion.dissimilarity_map(recon, feats)
  elif isinstance(criterion, nn.L1Loss):
    return (feats - recon).abs().mean(dim=1)
  elif isinstance(criterion, nn.HuberLoss):
    return _huber_elementwise(feats - recon, criterion.delta).mean(dim=1)
  else:
    # ค่า default / กรณี nn.MSELoss
    # Default / nn.MSELoss case.
    return ((feats - recon) ** 2).mean(dim=1)