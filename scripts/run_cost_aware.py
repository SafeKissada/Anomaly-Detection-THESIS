"""รันจาก artifact ที่มีอยู่แล้ว (scores_val.npz, scores_test.npz) ไม่ต้อง
train ใหม่

Run from existing artifacts (scores_val.npz, scores_test.npz) — no
re-training needed.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.config import Config
from src import io_utils
from src.cost_aware import cost_sweep_report


def main():
    """รัน cost-aware threshold sweep เต็มรูปแบบ: โหลด score ที่เซฟไว้ →
    sweep หลายค่า r → เซฟผลเป็น CSV

    ห่อด้วย main() (แทนที่จะเป็น module-level code ล้วน) ให้ตรงกับ
    pattern เดียวกับ scripts/train.py และ scripts/visualize.py — สำคัญ
    เพราะทำให้ import ไฟล์นี้จากที่อื่น (เช่น RUN.py) ได้โดยไม่รันทันที
    ตอน import เอง ต้องเรียก main() ตรงๆ ถึงจะทำงาน

    Run the full cost-aware threshold sweep: load saved scores → sweep
    several r values → save the results as a CSV.

    Wrapped in main() (instead of being plain module-level code) to
    match the same pattern as scripts/train.py and scripts/visualize.py
    — important because it lets this file be imported from elsewhere
    (e.g. RUN.py) without running immediately on import; main() must be
    called explicitly to actually do anything.
    """
    # ใช้ Config() instance เดียวกับ (หรือ config ที่ set CFG.SAVE_PATH ตรงกับ)
    # experiment ที่ต้องการวิเคราะห์ — io_utils ใช้ cfg.SAVE_PATH หา path ของ
    # scores_{split}.npz ต้องชี้ให้ตรง output directory ของ run ที่ต้องการ
    #
    # Use the same Config() instance (or one with CFG.SAVE_PATH pointed at)
    # the experiment you want to analyze — io_utils uses cfg.SAVE_PATH to
    # locate scores_{split}.npz, so this must point at the right run's
    # output directory.
    CFG = Config()

    # ใช้ io_utils.load_scores() แทนการ np.load() ไฟล์ดิบเอง — สำคัญ: key
    # ที่ถูกต้องสำหรับ ground-truth label คือ 'y_true' (int 0/1) ไม่ใช่
    # 'labels' (string เช่น "good"/"defect" — เป็นคนละตัวแปรกับ y_true ตาม
    # ที่นิยามไว้ใน io_utils.save_scores()) การเรียกผ่าน io_utils แบบนี้ยัง
    # กันไม่ให้ต้อง maintain loading logic ซ้ำซ้อนกับที่มีอยู่แล้วในไฟล์นั้น
    # ด้วย — ถ้า io_utils เปลี่ยน schema ในอนาคต จุดนี้จะตามไปเองอัตโนมัติ
    #
    # Uses io_utils.load_scores() instead of calling np.load() on the raw
    # file directly — important: the correct key for the ground-truth label
    # is 'y_true' (int 0/1), not 'labels' (a string like "good"/"defect" —
    # a completely different variable from y_true, as defined in
    # io_utils.save_scores()). Going through io_utils like this also avoids
    # duplicating the loading logic that already lives there — if io_utils'
    # schema changes later, this stays correct automatically.
    val  = io_utils.load_scores('val',  CFG)
    test = io_utils.load_scores('test', CFG)

    val_scores,  val_y  = val['scores'],  val['y_true']
    test_scores, test_y = test['scores'], test['y_true']

    r_values = [1, 5, 10, 20, 50, 100]
    report = cost_sweep_report(val_scores, val_y, test_scores, test_y, r_values)

    df = pd.DataFrame(report)
    print(df.to_string(index=False))

    output_csv = Path(CFG.SAVE_PATH) / 'cost_aware_sweep.csv'
    df.to_csv(output_csv, index=False)
    print(f'\nSaved -> {output_csv}')


if __name__ == '__main__':
    main()