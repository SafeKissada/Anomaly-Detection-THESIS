"""Convolutional autoencoder บน feature space (ไม่ใช่ pixel space) — encoder
บีบ feature ConvNeXt ที่ normalize แล้วลงไปที่ bottleneck แล้ว decoder
reconstruct กลับมา

Feature-space (not pixel-space) convolutional autoencoder — the encoder
compresses normalized ConvNeXt features down to a bottleneck, and the
decoder reconstructs them back.
"""
import torch
import torch.nn as nn


class LightCNBlock(nn.Module):
  """Block สไตล์ ConvNeXt แบบเบา: depthwise conv -> LayerNorm -> MLP (GELU)
  -> residual connection ใช้ทั้งใน encoder และ decoder เพื่อ refine
  feature ที่แต่ละ resolution level หลัง downsample/upsample

  A lightweight ConvNeXt-style block: depthwise conv -> LayerNorm -> MLP
  (GELU) -> residual connection. Used in both the encoder and decoder to
  refine features at each resolution level after downsampling/upsampling.
  """
  def __init__(self, dim, expand_ratio = 2, kernel_size = 7, drop_path = 0.0):
    super().__init__()
    hidden_dim = dim * expand_ratio
    self.dwconv = nn.Conv2d(
        in_channels    = dim,
        out_channels   = dim,
        kernel_size    = kernel_size,
        padding        = kernel_size // 2,
        groups         = dim,
        bias           = False)

    self.norm = nn.LayerNorm(dim, eps=1e-6)
    self.pwconv1 = nn.Linear(dim, hidden_dim)
    self.act     = nn.GELU()
    self.pwconv2 = nn.Linear(hidden_dim, dim)
    self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

  def forward(self, x):
    # x: [B, C, H, W]
    residual = x
    x = self.dwconv(x)
    x = x.permute(0, 2, 3, 1)  # [B, H, W, C] — LayerNorm/Linear ต้องการ channel เป็นแกนสุดท้าย / LayerNorm/Linear expect channel as the last axis
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)
    x = x.permute(0, 3, 1, 2)  # [B, C, H, W] — กลับไป layout เดิม / back to the original layout
    x = self.drop_path(x)
    return residual + x   # residual connection


def count_params(module):
  """นับจำนวน trainable parameter ทั้งหมดของ module

  Count the total number of trainable parameters in a module.
  """
  return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ConvEncoderWithCNBlock(nn.Module):
  """Encoder: downsample 3 ครั้ง (stride-2 conv แต่ละครั้ง) จาก in_ch ช่อง
  ไปถึง bottleneck ช่อง คั่นด้วย LightCNBlock เพื่อ refine feature ที่แต่ละ
  resolution level

  Encoder: 3 stride-2 downsample stages taking in_ch channels down to the
  bottleneck, each followed by a LightCNBlock to refine features at that
  resolution level.
  """
  def __init__(self, in_ch=576, bottleneck=64):
    super().__init__()
    mid1, mid2 = in_ch // 2, in_ch // 4 # 288, 144 ตามของเดิม / 288, 144 as before

    self.down1 = nn.Sequential(
        nn.Conv2d(in_ch, mid1, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(mid1),
        nn.GELU()
    )
    self.refine1 = LightCNBlock(mid1, expand_ratio=2)

    self.down2 = nn.Sequential(
        nn.Conv2d(mid1, mid2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(mid2),
        nn.GELU()
    )
    self.refine2 = LightCNBlock(mid2, expand_ratio=2)

    self.down3 = nn.Sequential(
        nn.Conv2d(mid2, bottleneck, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck),
        nn.GELU()
    )
    self.mid1, self.mid2 = mid1, mid2

  def forward(self, x):
    x = self.down1(x); x = self.refine1(x)
    x = self.down2(x); x = self.refine2(x)
    x = self.down3(x)
    return x


class ConvDecoderWithCNBlock(nn.Module):
  """Decoder: upsample 3 ครั้ง (transposed conv แต่ละครั้ง) จาก bottleneck
  กลับไปที่ out_ch ช่อง เป็นภาพสะท้อนของ ConvEncoderWithCNBlock — รับ
  mid1/mid2 มาจาก encoder โดยตรง (ผ่าน FeatureAutoencoder) เพื่อให้
  channel count ที่แต่ละ resolution level ตรงกันสมมาตร

  Decoder: 3 stride-2 upsample stages taking the bottleneck back up to
  out_ch channels — mirrors ConvEncoderWithCNBlock. Receives mid1/mid2
  directly from the encoder (via FeatureAutoencoder) so channel counts at
  each resolution level match symmetrically.
  """
  def __init__(
      self,
      out_ch         : int,
      bottleneck_ch  : int  = 64,
      mid1           : int  = None,
      mid2           : int  = None,
      cnblock_expand : int  = 2):
    super().__init__()

    self.mid2 = mid2 or max(out_ch // 4, bottleneck_ch * 2)
    self.mid1 = mid1 or max(out_ch // 2, bottleneck_ch * 4)

    self.up1 = nn.Sequential(
        nn.ConvTranspose2d(bottleneck_ch, self.mid2, kernel_size=3, stride=2,
                           padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(self.mid2),
        nn.GELU()
    )
    self.refine1 = LightCNBlock(self.mid2, expand_ratio=cnblock_expand)

    self.up2 = nn.Sequential(
        nn.ConvTranspose2d(self.mid2, self.mid1, kernel_size=4, stride=2,
                           padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(self.mid1),
        nn.GELU()
    )
    self.refine2 = LightCNBlock(self.mid1, expand_ratio=cnblock_expand)

    self.up3 = nn.Sequential(
        nn.ConvTranspose2d(self.mid1, out_ch, kernel_size=4, stride=2,
                           padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )
    self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)

  def forward(self, x):
    x = self.up1(x); x = self.refine1(x)
    x = self.up2(x); x = self.refine2(x)
    x = self.up3(x)
    x = self.proj(x)
    return x


class FeatureAutoencoder(nn.Module):
  """ประกอบ encoder + decoder เข้าด้วยกัน พร้อมเช็ค shape ให้แน่ใจว่า output
  ตรงกับ input เป๊ะ (ไม่งั้น loss ที่คำนวณต่อ element จะผิด shape ทันที)

  Wires the encoder + decoder together, with a shape check to guarantee
  the output exactly matches the input (otherwise any per-element loss
  would immediately break on a shape mismatch).
  """
  def __init__(
      self,
      feat_ch        : int,
      bottleneck_ch  : int = 64,
      encoder_cls    = ConvEncoderWithCNBlock,
      decoder_cls    = ConvDecoderWithCNBlock,
      encoder_kwargs : dict = None,
      decoder_kwargs : dict = None):
    super().__init__()

    encoder_kwargs = encoder_kwargs or {}
    decoder_kwargs = decoder_kwargs or {}

    self.encoder = encoder_cls(feat_ch, bottleneck_ch, **encoder_kwargs)
    self.decoder = decoder_cls(feat_ch, bottleneck_ch,
                               mid1 = self.encoder.mid1,
                               mid2 = self.encoder.mid2,
                               **decoder_kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x)
    recon = self.decoder(z)
    if recon.shape != x.shape:
      raise RuntimeError(
          f"FeatureAutoencoder: reconstructed shape {tuple(recon.shape)} != "
          f"input shape {tuple(x.shape)}. This almost always means "
          f"cfg.IMAGE_SIZE produces a backbone feature map whose spatial "
          f"size is not evenly divisible by 8 (3 stride-2 downsample "
          f"stages). Pick an IMAGE_SIZE such that "
          f"(IMAGE_SIZE / backbone_downsample_factor) is divisible by 8, "
          f"or reduce the number of downsample stages in "
          f"ConvEncoderWithCNBlock/ConvDecoderWithCNBlock accordingly.")
    return recon

  def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
    """ผ่านแค่ encoder เท่านั้น (ไม่ decode) — ใช้ตอนอยากรู้แค่ spatial size
    ของ bottleneck เช่นตอน train.py print ขนาดให้ดู ไม่ต้อง forward เต็ม

    Runs only the encoder (no decoding) — used when only the bottleneck's
    spatial size is needed, e.g. when train.py prints it, without a full
    forward pass.
    """
    return self.encoder(x)