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


def required_n_downsample(spatial_size: int) -> int:
  """หาค่า n_downsample ที่ "ถูกต้อง" สำหรับ feature map ขนาด spatial_size
  x spatial_size ก่อนจะยกไปสร้าง FeatureAutoencoder — **สำคัญ**: ไม่ใช่ทุกค่า
  n_downsample จะ round-trip กลับมาขนาดเดิมได้ ต้องใช้สูตรนี้เท่านั้น (เดา
  หรือใช้ "ลดลง 1" เฉยๆ ไม่ปลอดภัย — พิสูจน์แล้วว่าอาจ error หรือแย่กว่านั้น
  คือ shape ตรงกันโดยบังเอิญแต่ reconstruct ผิดจริง)

  เหตุผลเชิงคณิตศาสตร์: stage สุดท้าย (เข้า/ออกจาก bottleneck) ใช้
  kernel=3 ซึ่ง round-trip แม่นยำ (H -> ceil(H/2) -> H) ก็ต่อเมื่อ H ที่stage
  นั้นเป็นเลขคี่ ส่วน stage อื่นๆ ใช้ kernel=4 ซึ่ง round-trip แม่นยำ
  (H -> H/2 -> H) ก็ต่อเมื่อ H เป็นเลขคู่ ดังนั้น n_downsample ที่ถูกต้องคือ
  จำนวนครั้งที่ spatial_size หารด้วย 2 ลงตัวได้ (ก่อนเจอเลขคี่) บวก 1 —
  ตัวอย่าง: 28 หารด้วย 2 ได้ 2 ครั้ง (28->14->7, 7 เป็นเลขคี่) จึงต้อง
  n_downsample=3 (ตรงกับค่า default เดิมของโค้ดนี้พอดี)

  Finds the "correct" n_downsample for a spatial_size x spatial_size
  feature map before constructing a FeatureAutoencoder with it. **This
  matters**: not every n_downsample round-trips back to the original
  size — guessing or just "subtracting 1" is not safe (verified to
  either raise, or worse, silently produce the right shape via a
  different, non-inverse computation).

  Math: the final stage (into/out of the bottleneck) uses kernel=3,
  which round-trips exactly (H -> ceil(H/2) -> H) only when H at that
  stage is odd. Every other stage uses kernel=4, which round-trips
  exactly (H -> H/2 -> H) only when H is even. So the correct
  n_downsample is the number of times spatial_size divides evenly by 2
  (before hitting an odd number), plus 1 — e.g. 28 divides by 2 twice
  (28->14->7, 7 is odd) so n_downsample=3 (matching this module's
  original default exactly).
  """
  n, h = 0, spatial_size
  while h % 2 == 0:
    h //= 2
    n += 1
  return n + 1


class ConvEncoderWithCNBlock(nn.Module):
  """Encoder: downsample N ครั้ง (stride-2 conv แต่ละครั้ง) จาก in_ch ช่อง
  ไปถึง bottleneck ช่อง คั่นด้วย LightCNBlock เพื่อ refine feature ที่แต่ละ
  resolution level (ไม่ refine หลัง stage สุดท้ายที่เข้า bottleneck)

  N ถูกกำหนดผ่าน n_downsample (ค่าเดิม/ค่า default คือ 3 — พฤติกรรมและชื่อ
  attribute (down1/refine1/down2/refine2/down3) เหมือนโค้ดเดิมทุกประการเมื่อ
  n_downsample=3 เพื่อให้ checkpoint เก่าที่เทรนไว้โหลดกลับได้ปกติ — ใช้
  n_downsample น้อยกว่า 3 (เช่น 2) กับ feature map ที่ spatial size เล็ก
  ตั้งแต่ต้น (เช่น ConvNeXt Stage4 เดี่ยวๆ ที่ 7x7 @ IMAGE_SIZE=224) เพื่อไม่ให้
  ขนาดหลุดจนไม่ divisible — ดู FeatureAutoencoder ข้อมูลเพิ่มเติม

  Encoder: N stride-2 downsample stages taking in_ch channels down to the
  bottleneck, each followed by a LightCNBlock to refine features at that
  resolution level (no refine after the final stage into the bottleneck).

  N is set via n_downsample (default 3 — same behavior and attribute names
  (down1/refine1/down2/refine2/down3) as the original code when
  n_downsample=3, so existing trained checkpoints still load correctly).
  Use n_downsample < 3 (e.g. 2) with feature maps that are already small
  spatially (e.g. a lone ConvNeXt Stage4 at 7x7 @ IMAGE_SIZE=224) so the
  size doesn't shrink past being evenly divisible — see FeatureAutoencoder
  for more.
  """
  def __init__(self, in_ch: int, bottleneck: int = 64, n_downsample: int = 3):
    super().__init__()
    if n_downsample < 1:
      raise ValueError(f"n_downsample must be >= 1, got {n_downsample}")

    # channel ต่อ stage: in_ch -> in_ch//2 -> in_ch//4 -> ... -> bottleneck
    # (เหมือนเดิมเป๊ะที่ n_downsample=3: mid1=in_ch//2, mid2=in_ch//4)
    #
    # per-stage channels: in_ch -> in_ch//2 -> in_ch//4 -> ... -> bottleneck
    # (identical to the original at n_downsample=3: mid1=in_ch//2, mid2=in_ch//4)
    mids = [max(in_ch // (2 ** i), bottleneck) for i in range(1, n_downsample)]
    channels = [in_ch] + mids + [bottleneck]

    self.n_downsample = n_downsample
    for i in range(n_downsample):
      c_in, c_out = channels[i], channels[i + 1]
      is_last = (i == n_downsample - 1)
      # สอดคล้องกับพฤติกรรมเดิม: stage สุดท้ายที่เข้า bottleneck ใช้
      # kernel=3, stage อื่นๆ ใช้ kernel=4 (เหมือนโค้ดเดิมที่ down1/down2
      # ใช้ kernel=4 และ down3 (เข้า bottleneck) ใช้ kernel=3)
      #
      # matches the original: the final stage into the bottleneck uses
      # kernel=3, every other stage uses kernel=4 (mirrors the original
      # where down1/down2 used kernel=4 and down3, into the bottleneck,
      # used kernel=3)
      k = 3 if is_last else 4
      setattr(self, f"down{i+1}", nn.Sequential(
          nn.Conv2d(c_in, c_out, kernel_size=k, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(c_out),
          nn.GELU(),
      ))
      if not is_last:
        setattr(self, f"refine{i+1}", LightCNBlock(c_out, expand_ratio=2))

    self.mid1 = mids[0] if len(mids) >= 1 else bottleneck
    self.mid2 = mids[1] if len(mids) >= 2 else self.mid1

  def forward(self, x):
    for i in range(self.n_downsample):
      x = getattr(self, f"down{i+1}")(x)
      refine = getattr(self, f"refine{i+1}", None)
      if refine is not None:
        x = refine(x)
    return x


class ConvDecoderWithCNBlock(nn.Module):
  """Decoder: upsample N ครั้ง (transposed conv แต่ละครั้ง) จาก bottleneck
  กลับไปที่ out_ch ช่อง เป็นภาพสะท้อนของ ConvEncoderWithCNBlock — รับ
  mid1/mid2 มาจาก encoder โดยตรง (ผ่าน FeatureAutoencoder) เพื่อให้
  channel count ที่แต่ละ resolution level ตรงกันสมมาตร

  N ถูกกำหนดผ่าน n_downsample เหมือน ConvEncoderWithCNBlock (ต้องตรงกัน
  เสมอ — FeatureAutoencoder เป็นผู้ส่งค่าเดียวกันให้ทั้งคู่) พฤติกรรมและ
  ชื่อ attribute (up1/refine1/up2/refine2/up3/proj) เหมือนโค้ดเดิมทุก
  ประการเมื่อ n_downsample=3

  Decoder: N stride-2 upsample stages taking the bottleneck back up to
  out_ch channels — mirrors ConvEncoderWithCNBlock. Receives mid1/mid2
  directly from the encoder (via FeatureAutoencoder) so channel counts at
  each resolution level match symmetrically.

  N is set via n_downsample, same as ConvEncoderWithCNBlock (must always
  match — FeatureAutoencoder passes the same value to both). Behavior and
  attribute names (up1/refine1/up2/refine2/up3/proj) are identical to the
  original when n_downsample=3.
  """
  def __init__(
      self,
      out_ch         : int,
      bottleneck_ch  : int  = 64,
      mid1           : int  = None,
      mid2           : int  = None,
      cnblock_expand : int  = 2,
      n_downsample   : int  = 3):
    super().__init__()
    if n_downsample < 1:
      raise ValueError(f"n_downsample must be >= 1, got {n_downsample}")

    self.mid2 = mid2 or max(out_ch // 4, bottleneck_ch * 2)
    self.mid1 = mid1 or max(out_ch // 2, bottleneck_ch * 4)

    # ไล่ channel จาก bottleneck กลับไป out_ch เป็นภาพสะท้อนของ encoder —
    # ที่ n_downsample=3 คือ [bottleneck, mid2, mid1, out_ch] เป๊ะเหมือนเดิม
    #
    # channel progression from bottleneck back to out_ch, mirroring the
    # encoder — at n_downsample=3 this is exactly [bottleneck, mid2, mid1,
    # out_ch], same as the original.
    if n_downsample == 1:
      channels = [bottleneck_ch, out_ch]
    elif n_downsample == 2:
      channels = [bottleneck_ch, self.mid1, out_ch]
    else:
      mids_desc = [self.mid2, self.mid1]
      extra = n_downsample - 3
      if extra > 0:
        # เผื่อ n_downsample > 3 ในอนาคต: ขยาย channel แบบ geometric ระหว่าง
        # mid2 กับ bottleneck (กรณีนี้ยังไม่ถูกใช้งานจริงตอนนี้)
        #
        # in case n_downsample > 3 is needed in the future: interpolate
        # channels geometrically between mid2 and bottleneck (not
        # exercised by any variant currently planned)
        step = (self.mid2 - bottleneck_ch) / (extra + 1)
        mids_desc = [int(bottleneck_ch + step * (extra - j)) for j in range(extra)] + mids_desc
      channels = [bottleneck_ch] + mids_desc + [out_ch]

    self.n_downsample = n_downsample
    for i in range(n_downsample):
      c_in, c_out = channels[i], channels[i + 1]
      is_first = (i == 0)
      is_last = (i == n_downsample - 1)
      # สอดคล้องกับพฤติกรรมเดิม: stage แรกที่ออกจาก bottleneck ใช้
      # kernel=3 (มิเรอร์ down3 ของ encoder), stage อื่นๆ ใช้ kernel=4
      #
      # matches the original: the first stage out of the bottleneck uses
      # kernel=3 (mirrors the encoder's down3), every other stage uses
      # kernel=4
      k = 3 if is_first else 4
      setattr(self, f"up{i+1}", nn.Sequential(
          nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=2,
                             padding=1, output_padding=0, bias=False),
          nn.BatchNorm2d(c_out),
          nn.GELU(),
      ))
      if not is_last:
        setattr(self, f"refine{i+1}", LightCNBlock(c_out, expand_ratio=cnblock_expand))

    self.proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)

  def forward(self, x):
    for i in range(self.n_downsample):
      x = getattr(self, f"up{i+1}")(x)
      refine = getattr(self, f"refine{i+1}", None)
      if refine is not None:
        x = refine(x)
    x = self.proj(x)
    return x


class FeatureAutoencoder(nn.Module):
  """ประกอบ encoder + decoder เข้าด้วยกัน พร้อมเช็ค shape ให้แน่ใจว่า output
  ตรงกับ input เป๊ะ (ไม่งั้น loss ที่คำนวณต่อ element จะผิด shape ทันที)

  n_downsample กำหนดจำนวน stride-2 stage ของทั้ง encoder/decoder (ค่า
  default 3 คือพฤติกรรมเดิมทุกประการ) ต้องเลือกให้สอดคล้องกับ spatial size
  ของ feature ที่ extractor ส่งเข้ามา — ดูคำอธิบายเงื่อนไข divisibility ใน
  RuntimeError ด้านล่างถ้าเลือกไม่สอดคล้องกัน (เช่น ConvNeXt Stage4 เดี่ยวๆ
  ที่ spatial size เล็ก ต้องใช้ n_downsample=2 ไม่ใช่ 3)

  Wires the encoder + decoder together, with a shape check to guarantee
  the output exactly matches the input (otherwise any per-element loss
  would immediately break on a shape mismatch).

  n_downsample sets the number of stride-2 stages in both encoder/decoder
  (default 3 reproduces the original behavior exactly). Must be chosen to
  match the spatial size of the feature the extractor produces — see the
  divisibility explanation in the RuntimeError below if the choice doesn't
  fit (e.g. a lone ConvNeXt Stage4, whose spatial size is small, needs
  n_downsample=2, not 3).
  """
  def __init__(
      self,
      feat_ch        : int,
      bottleneck_ch  : int = 64,
      n_downsample   : int = 3,
      encoder_cls    = ConvEncoderWithCNBlock,
      decoder_cls    = ConvDecoderWithCNBlock,
      encoder_kwargs : dict = None,
      decoder_kwargs : dict = None):
    super().__init__()

    encoder_kwargs = encoder_kwargs or {}
    decoder_kwargs = decoder_kwargs or {}

    self.encoder = encoder_cls(feat_ch, bottleneck_ch,
                               n_downsample=n_downsample, **encoder_kwargs)
    self.decoder = decoder_cls(feat_ch, bottleneck_ch,
                               mid1 = self.encoder.mid1,
                               mid2 = self.encoder.mid2,
                               n_downsample = n_downsample,
                               **decoder_kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x)
    recon = self.decoder(z)
    if recon.shape != x.shape:
      n = self.encoder.n_downsample
      raise RuntimeError(
          f"FeatureAutoencoder: reconstructed shape {tuple(recon.shape)} != "
          f"input shape {tuple(x.shape)}. This almost always means the "
          f"backbone feature map's spatial size is not evenly divisible by "
          f"2**n_downsample (n_downsample={n} here, so needs to be "
          f"divisible by {2 ** n}). Either pick a cfg.IMAGE_SIZE that makes "
          f"(IMAGE_SIZE / backbone_downsample_factor) divisible by "
          f"{2 ** n}, or pass a smaller n_downsample to FeatureAutoencoder "
          f"that matches the feature map's spatial size (e.g. n_downsample="
          f"2 for a small feature map such as a lone ConvNeXt Stage4).")
    return recon

  def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
    """ผ่านแค่ encoder เท่านั้น (ไม่ decode) — ใช้ตอนอยากรู้แค่ spatial size
    ของ bottleneck เช่นตอน train.py print ขนาดให้ดู ไม่ต้อง forward เต็ม

    Runs only the encoder (no decoding) — used when only the bottleneck's
    spatial size is needed, e.g. when train.py prints it, without a full
    forward pass.
    """
    return self.encoder(x)