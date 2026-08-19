"""จุดเริ่มต้นการ visualize: โหลด artifact ที่ scripts/train.py เซฟไว้
(history.json, scores_{split}.npz, extractor_norm_stats.pt, threshold.json,
checkpoint .pth) แล้ว render กราฟ/heatmap ทั้งหมด

ไม่เทรนใหม่, ไม่คำนวณ score ใหม่, ไม่เรียก fit_normalization เลย ถ้า
artifact ที่ต้องใช้ตัวไหนหายไป จะ raise error บอกให้รัน train.py ก่อน

Visualization entry point: loads the artifacts saved by scripts/train.py
(history.json, scores_{split}.npz, extractor_norm_stats.pt, threshold.json,
checkpoint .pth) and renders all plots/heatmaps.

Does NOT retrain, re-score, or call fit_normalization. If any required
artifact is missing, it raises an error telling the user to run train.py first.
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ConvNeXtAutoencoder')

import pandas as pd
import torch
from sklearn.metrics import roc_curve
from rich.console import Console
from rich.panel import Panel

from config.config import Config
from src.evaluate import compute_metrics
from src import io_utils
from src import output_docs
from src.visual import (plot_class_distribution, plot_training_history,
                           plot_roc_curves, plot_pr_curves,
                           plot_confusion_matrices, plot_score_distributions,
                           visualize_heatmaps, browse_gallery,
                           gallery_original_images, gallery_processed_images,
                           gallery_preprocessed_images, gallery_preprocessed_overlay_images)

console = Console()


def main():
    CFG = Config()

    # Fail fast ถ้า train.py ยังไม่ได้สร้าง artifact ไว้
    # Fail fast if train.py has not produced the artifacts yet.
    io_utils.require_artifacts(CFG)

    # ── Load everything from disk / โหลดทุกอย่างจาก disk ─────────────────────
    history    = io_utils.load_history(CFG)
    thr_info   = io_utils.load_threshold(CFG)
    norm_stats = io_utils.load_norm_stats(CFG)
    ckpt       = torch.load(io_utils.checkpoint_path(CFG, io_utils.FINAL_CKPT_FILE),
                            map_location='cpu')

    threshold = float(thr_info['threshold'])
    data = {split: io_utils.load_scores(split, CFG) for split in io_utils.SPLITS}

    console.print(Panel(
        f'Backbone     : [cyan]ConvNeXt-{str(ckpt["backbone"]).capitalize()}[/cyan]\n'
        f'Feature dim  : [cyan]{norm_stats["out_channels"]} ch[/cyan]\n'
        f'Bottleneck   : [cyan]{ckpt["bottleneck"]} ch[/cyan]\n'
        f'Trained epochs : [cyan]{len(history["train_loss"])}[/cyan]\n'
        f'Threshold    : [cyan]{threshold:.6f}[/cyan]  '
        f'({thr_info["percentile"]:.0f}th pct of val normal)\n'
        f'Oracle (diag): [cyan]{thr_info["oracle_threshold"]:.6f}[/cyan]  '
        f'(F1={thr_info["oracle_f1"]:.4f})\n'
        f'Color mode   : [cyan]{CFG.COLOR_MODE}[/cyan]',
        title='[bold]Artifacts Loaded — rendering plots[/bold]'
    ))

    # ── Metrics from saved scores (no model inference) ─────────────────────
    # ── คำนวณ metric จาก score ที่เซฟไว้ (ไม่มีการรัน inference โมเดลใหม่เลย) ─
    metrics = {split: compute_metrics(d['scores'], d['y_true'], threshold)
               for split, d in data.items()}

    # ── ROC curve raw data → CSV แยก val/test (ตัวเลข → SAVE_PATH ไม่ใช่
    # OUTPUT_PATH เพื่อให้ตรง convention เดียวกับไฟล์อื่น) ───────────────
    # fpr/tpr/thresholds มาจาก sklearn.roc_curve() ตัวเดียวกับที่
    # plot_roc_curves() ใช้วาดกราฟ — เซฟจุดดิบไว้ให้เอาไปใช้ต่อได้โดยไม่
    # ต้องเปิด Python (เช่น เปิดใน Excel) หรือใช้เป็น input ของสคริปต์
    # multi-seed ROC aggregation ในอนาคต (vertical averaging ต้องใช้
    # fpr/tpr ดิบของแต่ละ seed ก่อน interpolate)
    #
    # Raw ROC curve data → separate CSV per val/test (numeric → SAVE_PATH,
    # not OUTPUT_PATH, matching the convention used for every other file).
    # fpr/tpr/thresholds come from the same sklearn.roc_curve() call that
    # plot_roc_curves() uses to draw the graph — the raw points are saved
    # so they can be used without opening Python (e.g. in Excel), or fed
    # into a future multi-seed ROC aggregation script (vertical averaging
    # needs each seed's raw fpr/tpr before interpolating).
    for split, m in metrics.items():
        fpr, tpr, roc_thresholds = roc_curve(data[split]['y_true'], data[split]['scores'])
        roc_df = pd.DataFrame({
            'fpr'      : fpr,
            'tpr'      : tpr,
            # roc_curve() คืน thresholds ยาวเท่า fpr/tpr เสมอ (จุดตัดที่
            # ทำให้ได้ fpr/tpr คู่นั้น) — เก็บไว้ด้วยเผื่อต้องย้อนกลับไป
            # หา threshold ที่จุดใดจุดหนึ่งบนเส้น
            # roc_curve() always returns thresholds with the same length
            # as fpr/tpr (the cut points that produce that fpr/tpr pair)
            # — kept in case you need to trace back to the threshold at
            # any specific point on the curve.
            'threshold': roc_thresholds,
        })
        roc_df.to_csv(f'{CFG.SAVE_PATH}/roc_curve_data_{split}.csv', index=False)
    print('ROC curve raw data (fpr/tpr/threshold) saved as CSV → SAVE_PATH.')

    # ── EDA: class distribution per split / สัดส่วน class ต่อ split ──────────
    plot_class_distribution({
        'Validation' : data['val']['labels'],
        'Test'       : data['test']['labels'],
    }, CFG)
    print('EDA chart saved.')

    # ── Training history / กราฟ training history ────────────────────────────
    save_file_path = plot_training_history(history, CFG)
    print(f'Training history plot saved successfully at: {save_file_path}')

    # ── ROC / PR / confusion matrices / score distributions ────────────────
    split_meta = [
        ('Validation Set', metrics['val'],   '#4CAF50'),
        ('Test Set',       metrics['test'],  '#FF5722'),
    ]
    plot_roc_curves(split_meta, CFG)
    plot_pr_curves(split_meta, CFG)
    plot_confusion_matrices(split_meta, CFG)
    plot_score_distributions(split_meta, threshold, CFG)

    # ── Heatmaps per split / heatmap ต่อ split ───────────────────────────────
    for split, split_name in [('val','Validation'), ('test','Test')]:
        d = data[split]
        visualize_heatmaps(
            d['paths'], d['orig_imgs'], d['heatmaps'], d['labels'],
            d['scores'], threshold, split_name, CFG, n_samples=20, image_kind='rgb')

        if CFG.COLOR_MODE != 'RGB':
            visualize_heatmaps(
                d['paths'], d['preproc_imgs'], d['heatmaps'], d['labels'],
                d['scores'], threshold, split_name, CFG, n_samples=20, image_kind='preproc')

    # ── Result gallery / gallery สรุปผล ──────────────────────────────────────
    split_arrays = {
        split: dict(paths=d['paths'], labels=d['labels'], scores=d['scores'],
                    hmaps=d['heatmaps'], imgs=d['orig_imgs'],
                    preproc_imgs=d['preproc_imgs'],
                    gt=metrics[split]['gt'], pred=metrics[split]['pred'])
        for split, d in data.items()
    }

    _rows = []
    for split_name, d in split_arrays.items():
        n = len(d['paths'])
        for i in range(n):
            _rows.append({
                'split'      : split_name,
                'idx_in_split': i,
                'filename'   : Path(d['paths'][i]).name,
                'path'       : d['paths'][i],
                'label_gt'   : d['labels'][i],
                'pred_label' : 'anomaly' if d['pred'][i] == 1 else 'normal',
                'score'      : float(d['scores'][i]),
                'correct'    : bool(d['gt'][i] == d['pred'][i]),
            })

    df_gallery = pd.DataFrame(_rows)
    # gallery_index.csv เป็นตัวเลข/tabular (ไม่ใช่ภาพ) จึงเก็บที่ SAVE_PATH
    # แทน OUTPUT_PATH — OUTPUT_PATH หลังจากนี้จะมีแต่ไฟล์ .png เท่านั้น
    #
    # gallery_index.csv is numeric/tabular (not an image), so it's saved
    # under SAVE_PATH instead of OUTPUT_PATH — from this point on,
    # OUTPUT_PATH contains only .png files.
    df_gallery.to_csv(f'{CFG.SAVE_PATH}/gallery_index.csv', index=False)

    print(f'df_gallery: {len(df_gallery):,} แถว '
          f"(val={len(data['val']['paths']):,}, "
          f"test={len(data['test']['paths']):,})")

    print('\nจำนวนภาพต่อ split × label (ground truth):')
    print(df_gallery.groupby(['split', 'label_gt']).size().unstack(fill_value=0))

    print('\nจำนวนภาพต่อ split × ผลทาย (ถูก/ผิด):')
    print(df_gallery.groupby(['split', 'correct']).size().unstack(fill_value=0))

    # gallery 3 คอลัมน์ (ภาพต้นฉบับ | error map | overlay) — เฉพาะ sample ที่ทายผิด
    # 3-column gallery (original | error map | overlay) — misclassified samples
    _ = browse_gallery(df_gallery, split_arrays, CFG, split='test', correct=False, n=100)
    _ = browse_gallery(df_gallery, split_arrays, CFG, split='val',  correct=False, n=100)

    # ── Output gallery แบบที่ 1: ภาพจริง (RGB, always) ──────────────────────
    # ── Output gallery แบบที่ 2: ภาพหลัง image processing (heatmap overlay on RGB) ─
    for split_name in ['val', 'test']:
        _ = gallery_original_images(
            df_gallery, split_arrays, CFG,
            split=split_name, n=20, ncols=5)
        _ = gallery_processed_images(
            df_gallery, split_arrays, CFG,
            split=split_name, n=20, ncols=5)

    if CFG.COLOR_MODE != 'RGB':
        print(f"\nCOLOR_MODE = '{CFG.COLOR_MODE}' → เพิ่ม gallery เวอร์ชัน preprocessed ด้วย")
        for split_name in ['val', 'test']:
            _ = gallery_preprocessed_images(
                df_gallery, split_arrays, CFG,
                split=split_name, n=20, ncols=5)
            _ = gallery_preprocessed_overlay_images(
                df_gallery, split_arrays, CFG,
                split=split_name, n=20, ncols=5)

    # เขียน README.md อธิบายไฟล์ภาพทุกตัวใน OUTPUT_PATH แบบ dynamic (เช็ค
    # ไฟล์ที่มีอยู่จริง ณ ตอนนี้เท่านั้น) — ไม่แตะ SAVE_PATH เลย เพราะ
    # visualize.py ไม่เคยเขียน checkpoint/metric/score ลงนั้น (มีแค่
    # gallery_index.csv ที่ย้ายมาแล้วข้างบน)
    #
    # Writes a README.md documenting every image file in OUTPUT_PATH
    # dynamically (only files that exist right now) — never touches
    # SAVE_PATH, since visualize.py never writes checkpoints/metrics/
    # scores there (except gallery_index.csv, already moved above).
    readme_path = output_docs.write_output_path_readme(CFG)
    print(f'\nWrote documentation -> {readme_path}')

    logger.info('All plots/heatmaps rendered.')


if __name__ == '__main__':
    main()