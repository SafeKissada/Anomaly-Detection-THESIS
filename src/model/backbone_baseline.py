"""ตัวสกัด feature ConvNeXt แบบ frozen (Stage2+Stage3)

Frozen ConvNeXt feature extractor (Stage2+Stage3).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger('ConvNeXtAutoencoder')


class ConvNeXtExtractor(nn.Module):
  """ConvNeXt ที่ pretrain มาแล้วและ frozen ทั้งตัว (ไม่มี parameter ไหน
  trainable เลย) สกัด feature จาก Stage2 กับ Stage3 แล้วต่อกัน (concat)
  เป็น output เดียว — Stage3 ถูก upsample ด้วย bilinear interpolation ให้
  spatial size เท่ากับ Stage2 ก่อน concat

  A fully pretrained, frozen ConvNeXt (no trainable parameters at all)
  that extracts Stage2 and Stage3 features and concatenates them into one
  output — Stage3 is bilinearly upsampled to Stage2's spatial size before
  concatenation.
  """
  _VARIANTS = {
      'tiny' : (models.convnext_tiny,  models.ConvNeXt_Tiny_Weights.DEFAULT),
      'small': (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
      'base' : (models.convnext_base,  models.ConvNeXt_Base_Weights.DEFAULT),
      'large': (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
  }

  def __init__(self, variant: str = 'tiny'):
    super().__init__()
    assert variant in self._VARIANTS, f"Unknown variant: {variant}"

    model_fn, weights  = self._VARIANTS[variant]
    backbone           = model_fn(weights=weights)

    self.stem = backbone.features[0]
    self.stage1 = backbone.features[1]
    self.stage2 = nn.Sequential(
        backbone.features[2],
        backbone.features[3])
    self.stage3 = nn.Sequential(
        backbone.features[4],
        backbone.features[5])

    # แช่แข็งทุก parameter — extractor นี้ไม่ถูกเทรนเพิ่มเลย
    # Freeze every parameter — this extractor is never further trained.
    for param in self.parameters():
      param.requires_grad = False

    self.eval()

    # ทำ dummy forward pass เพื่อหา out_channels/spatial_size จริง
    # แทนที่จะ hardcode ไว้ (เผื่อ ConvNeXt variant ต่างกัน shape ต่างกัน)
    # Do a dummy forward pass to discover the real out_channels/
    # spatial_size instead of hardcoding them (different ConvNeXt variants
    # can have different shapes).
    with torch.no_grad():
      dummy = torch.zeros(1, 3, 224, 224)
      feat2, feat3 = self._extract_features(dummy)
      self.out_channels = feat2.shape[1] + feat3.shape[1]
      self.spatial_size = feat2.shape[-2:]

    self.register_buffer('feat_mean', torch.zeros(self.out_channels, 1, 1))
    self.register_buffer('feat_std',  torch.ones(self.out_channels, 1, 1))
    self.norm_fitted = False
    print(f'ConvNeXt-{variant.capitalize()} Extractor (pretrained, frozen) → '
          f'{self.out_channels} ch, spatial {tuple(self.spatial_size)}')

  def train(self, mode: bool = True):
    # บังคับให้อยู่ใน eval mode เสมอ ไม่ว่าจะเรียก .train() จากไหนก็ตาม
    # เพราะ extractor นี้ frozen ตลอด ไม่มีทาง train ได้จริง (กัน BatchNorm/
    # Dropout เปลี่ยนพฤติกรรมโดยไม่ตั้งใจถ้ามีคนเผลอเรียก ae.train() แล้ว
    # ลาม module ย่อยไปด้วย)
    #
    # Always stay in eval mode regardless of who calls .train(), since
    # this extractor is permanently frozen and can never actually be
    # trained (guards against BatchNorm/Dropout accidentally changing
    # behavior if something upstream calls .train() and it propagates
    # down to this submodule).
    super().train(False)
    return self

  def _extract_features(self, x: torch.Tensor):
    x = self.stem(x)
    x = self.stage1(x)
    feat2 = self.stage2(x)
    feat3 = self.stage3(feat2)
    return feat2, feat3

  @torch.no_grad()
  def fit_normalization(self, loader, device, max_batches: int = None):
    """คำนวณค่า mean/std ต่อ channel จาก loader ที่ให้มา (ควรเป็นภาพ
    ปกติ/good เท่านั้น) แล้วเซฟไว้ใน feat_mean/feat_std เพื่อใช้ normalize()
    ทีหลัง — ต้องเรียกครั้งเดียวก่อนเริ่มเทรน

    Compute per-channel mean/std from the given loader (should contain
    normal/good images only) and store them in feat_mean/feat_std for
    later use by normalize() — call once before training starts.
    """
    self.eval()
    sum_   = torch.zeros(self.out_channels, device=device)
    sumsq  = torch.zeros(self.out_channels, device=device)
    n_pix  = 0
    for bi, (norm_t, _, _, _, _, _) in enumerate(loader):
      if max_batches is not None and bi >= max_batches:
        break
      feat = self.forward(norm_t.to(device))            # [B, C, H, W]
      b, _, h, w = feat.shape
      sum_   += feat.sum(dim=(0, 2, 3))
      sumsq  += (feat ** 2).sum(dim=(0, 2, 3))
      n_pix  += b * h * w
    mean = sum_ / n_pix
    var  = (sumsq / n_pix - mean ** 2).clamp(min=1e-8)
    self.feat_mean = mean.view(-1, 1, 1)
    self.feat_std  = var.sqrt().view(-1, 1, 1)
    self.norm_fitted = True
    return self.feat_mean, self.feat_std

  def normalize(self, feat: torch.Tensor) -> torch.Tensor:
    """z-score normalize feature map ด้วยค่า mean/std ที่ fit ไว้แล้ว

    z-score normalize a feature map using the already-fitted mean/std.
    """
    if not self.norm_fitted:
      logger.warning('normalize() called before fit_normalization() — '
                     'features are NOT scale-normalized yet.')
    return (feat - self.feat_mean) / self.feat_std

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feat2, feat3 = self._extract_features(x)
    feat3 = F.interpolate(
        feat3,
        size=feat2.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    return torch.cat([feat2, feat3], dim=1)