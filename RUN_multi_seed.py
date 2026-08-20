"""รัน pipeline เดียวกับ RUN.py (train → visualize → cost-aware) ซ้ำ
หลาย seed ต่อเนื่องกัน (multi-seed) สำหรับ config ที่ชนะจาก OFAT แล้ว
(ตามที่ตกลงกันไว้ — ไม่ต้อง multi-seed ทุก config)

**Reuse `OVERRIDES` เดียวกับ RUN.py ทุกประการ** — import RUN.py ตรงๆ
แทนที่จะ copy dict OVERRIDES มาซ้ำอีกชุด กัน 2 ไฟล์ไม่ sync กัน

**การตัดสินใจสำคัญที่ต้องเขียนกำกับไว้ในเล่ม**: ไฟล์นี้แยก
`SPLIT_CACHE_PATH` ตาม seed ด้วย (ไม่ใช่แค่ SAVE_PATH/OUTPUT_PATH) —
เป็นการตัดสินใจที่ตั้งใจให้ seed คุมทั้งการแบ่ง train/val/test และ
training randomness พร้อมกัน ผลคือ **variance ที่วัดได้จาก multi-seed
เป็น variance รวมทั้งสองแหล่ง** (การแบ่งข้อมูลต่างกัน + training
randomness) ไม่ใช่ training randomness ล้วนๆ แบบที่งานส่วนใหญ่ทำกัน —
ถ้าต้องการ isolate เฉพาะ training randomness (แชร์ split เดียวกันทุก
seed) ให้ตั้ง TEMPLATE_KEYS ด้านล่างเอา 'SPLIT_CACHE_PATH' ออก

Runs the same pipeline as RUN.py (train → visualize → cost-aware)
repeatedly across several seeds, for a config that already won the OFAT
round (per the earlier decision — multi-seed is not run for every
config).

**Reuses the exact same `OVERRIDES` as RUN.py** — imports RUN.py
directly instead of copying the OVERRIDES dict a second time, so the two
files can never drift out of sync.

**Important decision to document in the thesis**: this file also splits
`SPLIT_CACHE_PATH` per seed (not just SAVE_PATH/OUTPUT_PATH) — a
deliberate choice to let seed control both the train/val/test split and
training randomness together. As a result, **the variance measured by
multi-seed here is the combined variance from both sources** (different
data splits + training randomness), not training randomness alone as
most work does — if you want to isolate training randomness only (share
the same split across all seeds), remove 'SPLIT_CACHE_PATH' from
TEMPLATE_KEYS below.

**สำคัญ**: import RUN จะรัน module-level code ของ RUN.py (สร้าง
OVERRIDES + monkey-patch Config.__init__) แต่ "if __name__ ==
'__main__':" ข้างในจะไม่ทำงาน (import ≠ run ตรงๆ) ปลอดภัย ไม่รัน
pipeline ซ้อนโดยไม่ตั้งใจ
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import RUN  

SEEDS = [1, 14, 42, 63, 123, 228, 450, 1357, 2512 , 19999]
TEMPLATE_KEYS = ['SPLIT_CACHE_PATH', 'SAVE_PATH', 'OUTPUT_PATH']
_current_seed = RUN.OVERRIDES['SEED']
_marker = f'SEED {_current_seed}'
_placeholder = 'SEED {seed}'

path_templates = {}
for key in TEMPLATE_KEYS:
    original_value = RUN.OVERRIDES[key]
    if _marker not in original_value:
        raise ValueError(
            f"ไม่เจอ '{_marker}' ใน RUN.OVERRIDES['{key}'] (= {original_value!r}) "
            f"— ต้องมีข้อความ '{_marker}' อยู่ในนั้นให้ script แทนที่ด้วย seed อื่นได้ "
            f"ถ้าตั้งชื่อโฟลเดอร์ต่างจากนี้ (เช่น เว้นวรรคไม่เหมือนกัน) ให้แก้ให้ตรงก่อน "
            f"หรือแก้ TEMPLATE_KEYS/_marker ด้านบนของไฟล์นี้เอง")
    path_templates[key] = original_value.replace(_marker, _placeholder)

print("Path templates ที่ตรวจพบ (จะแทน {seed} ด้วยเลข seed แต่ละรอบ):")
for key, tmpl in path_templates.items():
    print(f"  {key} = {tmpl!r}")

results_log = []

for i, seed in enumerate(SEEDS, start=1):
    print(f"\n{'=' * 70}")
    print(f" MULTI-SEED RUN [{i}/{len(SEEDS)}] — SEED={seed}")
    print(f"{'=' * 70}")

    RUN.OVERRIDES['SEED'] = seed
    for key, tmpl in path_templates.items():
        RUN.OVERRIDES[key] = tmpl.format(seed=seed)

    print(f"  SPLIT_CACHE_PATH -> {RUN.OVERRIDES['SPLIT_CACHE_PATH']}")
    print(f"  SAVE_PATH        -> {RUN.OVERRIDES['SAVE_PATH']}")
    print(f"  OUTPUT_PATH      -> {RUN.OVERRIDES['OUTPUT_PATH']}")

    try:
        RUN.train.main()
        RUN.visualize.main()
        if RUN.RUN_COST_AWARE_ANALYSIS:
            RUN.run_cost_aware.main()
        results_log.append((seed, 'OK', None))
        print(f"\n✅ seed={seed} เสร็จสมบูรณ์ -> {RUN.OVERRIDES['SAVE_PATH']}")

    except Exception as e:
        results_log.append((seed, 'FAILED', str(e)))
        print(f"\n❌ seed={seed} ล้มเหลว: {e}")
        traceback.print_exc()
        print("ข้าม seed นี้ ไปทำ seed ถัดไปต่อ...")
        continue

print(f"\n{'=' * 70}")
print(" สรุปผล Multi-Seed Run")
print(f"{'=' * 70}")
for seed, status, err in results_log:
    line = f"  seed={seed:<4}  {status}"
    if err:
        line += f"  ({err})"
    print(line)

n_ok = sum(1 for _, s, _ in results_log if s == 'OK')
print(f"\nสำเร็จ {n_ok}/{len(SEEDS)} seed")
if n_ok < len(SEEDS):
    print("⚠️  มี seed ที่ล้มเหลว — เช็ค traceback ด้านบนก่อนเอาผลไปสรุปสถิติ "
         "(mean_auc/std_auc ต้องคำนวณจากเฉพาะ seed ที่ OK เท่านั้น)")