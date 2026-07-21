"""Sanity check สำหรับ SSIM loss ก่อนรัน scripts/train.py เต็มรูปแบบ

ทำไมต้องเช็ค:
  SSIMLoss (src/losses.py) ใช้ค่าคงที่ C1=0.01^2, C2=0.03^2 ซึ่งออกแบบมา
  โดยสมมติว่าอินพุตมี dynamic range เป็นบวก (~[0,1] แบบภาพ) แต่ในโค้ดนี้
  SSIM ถูกคำนวณบน "feature map ที่ normalize แบบ z-score" (mean~0, std~1,
  ค่าติดลบได้) จาก extractor.normalize() ไม่ใช่ภาพตรงๆ
  -> luminance term (2*mu_x*mu_y + C1) อาจติดลบ/พฤติกรรมไม่ตรงทฤษฎี
  สคริปต์นี้โหลด extractor + 1 batch จริงจาก TRAIN_DIR แล้วเช็คว่า:
    1. SSIM loss/map มี NaN หรือ Inf หรือไม่
    2. ค่าที่ได้อยู่ในช่วงที่พอสมเหตุสมผลหรือไม่
       (loss ควรใกล้ 0 เมื่อ recon == feat, ควรเพิ่มขึ้นเมื่อใส่ noise)
    3. เทียบ scale ของ SSIM loss กับ MSE loss บนอินพุตชุดเดียวกัน
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config.config import Config, set_seed
from src.data.dataset import scan_directory, build_transforms, AnomalyDataset, make_loader
from src.model.backbone_baseline import ConvNeXtExtractor
from src.losses import SSIMLoss
import torch.nn as nn


def section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_tensor(name: str, t: torch.Tensor):
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    status = "OK" if (n_nan == 0 and n_inf == 0) else "FAIL"
    print(f"[{status}] {name:20s} shape={tuple(t.shape)}  "
          f"min={t.min().item(): .6f}  max={t.max().item(): .6f}  "
          f"mean={t.mean().item(): .6f}  nan={n_nan}  inf={n_inf}")
    return n_nan == 0 and n_inf == 0


def main():
    CFG = Config()
    set_seed(CFG.SEED)

    section("1) โหลด extractor + 1 batch จริงจาก TRAIN_DIR")
    df_train = scan_directory(CFG.TRAIN_DIR, CFG)
    if len(df_train) == 0:
        print("ไม่พบไฟล์ภาพใน CFG.TRAIN_DIR — ตรวจสอบ path ใน config.py ก่อน")
        return

    imagenet_tf, _, _, _ = build_transforms(CFG)
    ds = AnomalyDataset(df_train, imagenet_tf, imagenet_tf, CFG.IMAGE_SIZE)
    loader = make_loader(ds, CFG, shuffle=True)

    norm_t, _, _, paths, labels, _ = next(iter(loader))
    print(f"batch size = {norm_t.size(0)}  ตัวอย่าง path[0] = {paths[0]}")

    extractor = ConvNeXtExtractor(variant=CFG.BACKBONE).to(CFG.DEVICE)
    norm_t = norm_t.to(CFG.DEVICE)

    with torch.no_grad():
        feats_raw = extractor(norm_t)
    ok1 = check_tensor("raw features (pre-norm)", feats_raw)

    # fit normalization บน batch นี้เอง แค่เพื่อทดสอบ (ของจริงต้อง fit บน
    # normal_loader ทั้งชุดใน train.py ปกติ)
    extractor.fit_normalization(loader, CFG.DEVICE, max_batches=5)
    with torch.no_grad():
        feats = extractor.normalize(extractor(norm_t))
    ok2 = check_tensor("z-score normalized features", feats)

    section("2) ทดสอบ SSIMLoss บนฟีเจอร์จริง (recon == feats, ควร loss ~ 0)")
    ssim_loss = SSIMLoss().to(CFG.DEVICE)
    with torch.no_grad():
        loss_identical = ssim_loss(feats, feats)
        map_identical = ssim_loss.dissimilarity_map(feats, feats)
    ok3 = check_tensor("SSIM loss (identical)", loss_identical.unsqueeze(0))
    ok4 = check_tensor("SSIM dissim map (identical)", map_identical)
    print(f"-> คาดหวัง: loss ควรใกล้ 0 มากๆ (ทน tolerance ตัวเลข)  ได้จริง = {loss_identical.item():.8f}")

    section("3) ทดสอบ SSIMLoss เมื่อเพิ่ม noise ทีละระดับ (loss ควรเพิ่มขึ้นตาม noise)")
    mse_loss = nn.MSELoss()
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
    prev_ssim = -1.0
    monotonic_ok = True
    for sigma in noise_levels:
        noisy = feats + sigma * torch.randn_like(feats)
        with torch.no_grad():
            l_ssim = ssim_loss(noisy, feats).item()
            l_mse = mse_loss(noisy, feats).item()
        print(f"  noise_std={sigma:4.1f}  ->  SSIM_loss={l_ssim: .6f}   MSE_loss={l_mse: .6f}")
        if l_ssim < prev_ssim - 1e-6:
            monotonic_ok = False
        prev_ssim = l_ssim
    print(f"-> SSIM loss เพิ่มขึ้นตาม noise (monotonic) = {monotonic_ok}")

    section("4) ทดสอบด้วย recon จาก decoder จริงแบบสุ่ม (untrained) เทียบ scale SSIM vs MSE")
    from src.model.autoencoder import FeatureAutoencoder
    ae = FeatureAutoencoder(feat_ch=extractor.out_channels,
                             bottleneck_ch=CFG.AE_BOTTLENECK_CH).to(CFG.DEVICE)
    with torch.no_grad():
        recon = ae(feats)
    ok5 = check_tensor("recon (untrained AE)", recon)
    with torch.no_grad():
        l_ssim_untrained = ssim_loss(recon, feats).item()
        l_mse_untrained = mse_loss(recon, feats).item()
    print(f"  Untrained AE  ->  SSIM_loss={l_ssim_untrained:.6f}   MSE_loss={l_mse_untrained:.6f}")

    section("สรุปผล")
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5 and monotonic_ok
    if all_ok:
        print("[PASS] ไม่พบ NaN/Inf และ SSIM loss มีพฤติกรรม monotonic ตาม noise "
              "-> น่าจะปลอดภัยที่จะไปรัน scripts/train.py ด้วย CFG.LOSS='SSIM' ต่อได้")
    else:
        print("[WARN] พบความผิดปกติอย่างน้อย 1 จุดด้านบน (NaN/Inf หรือ loss ไม่ monotonic) "
              "-> ควรตรวจสอบก่อนรันเทรนเต็ม 150 epochs เช่น ลองปรับ window_size/sigma "
              "ใน SSIMLoss หรือพิจารณาใช้ 'SSIM_MSE' (CombinedLoss) แทนถ้า SSIM เดี่ยวไม่เสถียร")


if __name__ == "__main__":
    main()