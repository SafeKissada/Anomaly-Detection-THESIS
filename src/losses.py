import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
  """Per-pixel cosine distance across the channel axis.

  Reconstruct feature maps live in (B, C, H, W); at each spatial position
  (h, w) the C-length vector is treated as one direction. Scale-invariant
  by construction, so it needs no data_range/luminance calibration and is
  stationary across training regardless of how the z-scored feature
  values drift (unlike a pixel-space SSIM adapted to feature space would
  be -- see loss_functions_summary.md for why SSIM was removed).

  Caveat: scale-invariant by construction -- alone it never penalizes the
  decoder for drifting the output norm away from the input's, only the
  direction. Prefer CosineMSELoss unless that's specifically what you want.
  """

  def __init__(self, eps: float = 1e-8):
    super().__init__()
    self.eps = eps

  def _cosine_distance_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_sim = F.cosine_similarity(recon, target, dim=1, eps=self.eps)  # (B, H, W)
    return 1.0 - cos_sim

  def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return self._cosine_distance_map(recon, target).mean()

  def dissimilarity_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return self._cosine_distance_map(recon, target)


class CosineMSELoss(nn.Module):
  """Cosine distance (direction / cross-channel activation pattern) combined
  with MSE (magnitude) -- the MSE term is what keeps CosineLoss's scale
  invariance from letting the decoder's output norm drift unpenalized.

  lam=1.0 reduces to pure cosine, lam=0.0 reduces to pure MSE.
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
    cos_map = self.cosine._cosine_distance_map(recon, target)  # (B, H, W)
    mse_map = ((target - recon) ** 2).mean(dim=1)              # (B, H, W)
    return self.lam * cos_map + (1.0 - self.lam) * mse_map


def _huber_elementwise(diff: torch.Tensor, delta: float) -> torch.Tensor:
  """Per-element Huber loss (before any reduction), matching nn.HuberLoss's
  own formula exactly:
      0.5 * diff**2            if |diff| <  delta   (quadratic, like MSE)
      delta * (|diff| - 0.5*delta)  if |diff| >= delta   (linear, like MAE)
  Factored out here so both elementwise_error_map() (below) and
  process_single_heatmap() (src/engine.py) compute it identically instead
  of maintaining two copies of the same formula.
  """
  abs_diff = diff.abs()
  quadratic = 0.5 * diff ** 2
  linear = delta * (abs_diff - 0.5 * delta)
  return torch.where(abs_diff < delta, quadratic, linear)


def get_criterion(cfg) -> nn.Module:
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
  """Per-pixel error map used for anomaly scoring — MUST match the
  reduction the given `criterion` actually trains on, or the score/AUROC/
  threshold reported would silently be computed from a different error
  metric than the one the model was optimized for.
  """
  if isinstance(criterion, (CosineLoss, CosineMSELoss)):
    return criterion.dissimilarity_map(recon, feats)
  elif isinstance(criterion, nn.L1Loss):
    return (feats - recon).abs().mean(dim=1)
  elif isinstance(criterion, nn.HuberLoss):
    return _huber_elementwise(feats - recon, criterion.delta).mean(dim=1)
  else:
    # Default / nn.MSELoss case.
    return ((feats - recon) ** 2).mean(dim=1)