import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
  def __init__(self, window_size: int = 7, sigma: float = 1.5):
    super().__init__()
    self.window_size = window_size
    self.sigma = sigma
    self.register_buffer('window', self._make_window(window_size, sigma), persistent=False)

  @staticmethod
  def _gaussian_kernel1d(window_size, sigma):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

  def _make_window(self, window_size, sigma):
    g1d = self._gaussian_kernel1d(window_size, sigma)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    return g2d.unsqueeze(0).unsqueeze(0)
  
  def _rescale_to_unit_range(self, x, y):
    with torch.no_grad():
        flat = torch.cat([x, y], dim=1)
        dims = tuple(range(1, flat.dim()))
        lo = flat.amin(dim=dims, keepdim=True)
        hi = flat.amax(dim=dims, keepdim=True)
        data_range = (hi - lo).clamp(min=1e-6)
    x01 = ((x - lo) / data_range).clamp(0.0, 1.0)
    y01 = ((y - lo) / data_range).clamp(0.0, 1.0)
    return x01, y01

  def _ssim_map(self, x, y):
    x, y = self._rescale_to_unit_range(x, y)
    c = x.shape[1]
    window = self.window.to(device=x.device, dtype=x.dtype).expand(c, 1, self.window_size, self.window_size)
    pad = self.window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map

  def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ssim_map = self._ssim_map(recon, target)
    return 1.0 - ssim_map.mean()

  def dissimilarity_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ssim_map = self._ssim_map(recon, target)
    return (1.0 - ssim_map).mean(dim=1)


class CombinedLoss(nn.Module):
  def __init__(self, alpha: float = 0.5, beta: float = 0.5,
               window_size: int = 7, sigma: float = 1.5):
    super().__init__()
    self.alpha = alpha
    self.beta  = beta
    self.ssim  = SSIMLoss(window_size=window_size, sigma=sigma)
    self.mse   = nn.MSELoss()

  def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss_ssim = self.ssim(recon, target)
    loss_mse  = self.mse(recon, target)
    return self.alpha * loss_ssim + self.beta * loss_mse

  def dissimilarity_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ssim_map = self.ssim.dissimilarity_map(recon, target)          # [B, H, W]
    mse_map  = ((target - recon) ** 2).mean(dim=1)                 # [B, H, W]
    return self.alpha * ssim_map + self.beta * mse_map


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
  if loss_name == 'SSIM':
    return SSIMLoss()
  elif loss_name == 'MSE':
    return nn.MSELoss()
  elif loss_name in ('MAE', 'L1'):
    return nn.L1Loss()
  elif loss_name in ('HUBER', 'SMOOTH_L1', 'SMOOTHL1'):
    return nn.HuberLoss(delta=cfg.HUBER_DELTA)
  elif loss_name in ('SSIM_MSE', 'SSIM+MSE', 'SSIMMSE'):
    return CombinedLoss(alpha=cfg.SSIM_WEIGHT, beta=cfg.MSE_WEIGHT)
  else:
    raise ValueError(f"Unknown cfg.LOSS: {cfg.LOSS!r} (expected 'MSE', 'MAE', "
                     f"'HUBER', 'SSIM', or 'SSIM_MSE')")


def elementwise_error_map(feats: torch.Tensor, recon: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
  """Per-pixel error map used for anomaly scoring — MUST match the
  reduction the given `criterion` actually trains on, or the score/AUROC/
  threshold reported would silently be computed from a different error
  metric than the one the model was optimized for.
  """
  if isinstance(criterion, (SSIMLoss, CombinedLoss)):
    return criterion.dissimilarity_map(recon, feats)
  elif isinstance(criterion, nn.L1Loss):
    return (feats - recon).abs().mean(dim=1)
  elif isinstance(criterion, nn.HuberLoss):
    return _huber_elementwise(feats - recon, criterion.delta).mean(dim=1)
  else:
    # Default / nn.MSELoss case.
    return ((feats - recon) ** 2).mean(dim=1)
