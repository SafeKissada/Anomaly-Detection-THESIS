import torch
import torch.nn as nn


class LightCNBlock(nn.Module):
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
    x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)
    x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
    x = self.drop_path(x)
    return residual + x   # residual connection


def count_params(module):
  return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ConvEncoderWithCNBlock(nn.Module):
  def __init__(self, in_ch=576, bottleneck=64):
    super().__init__()
    mid1, mid2 = in_ch // 2, in_ch // 4 # 288, 144 ตามของเดิม

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
      # The encoder/decoder kernel/stride/padding combination (k4s2p1 x2 +
      # k3s2p1, mirrored on the decoder side) only reproduces the exact
      # input spatial size for feature maps whose H and W are divisible by
      # 8 (true for the default IMAGE_SIZE=224 -> 28x28 ConvNeXt-tiny stage2
      # feature map, but NOT guaranteed for other IMAGE_SIZE values). Rather
      # than letting this surface later as a cryptic broadcasting error
      # inside the loss function, fail immediately here with a clear,
      # actionable message.
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
    return self.encoder(x)
