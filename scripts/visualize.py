"""Visualization entry point: loads the artifacts saved by scripts/train.py
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

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel

from config.config import Config
from src.evaluate import compute_metrics
from src import io_utils
from src.visual import (plot_class_distribution, plot_training_history,
                           plot_roc_curves, plot_pr_curves,
                           plot_confusion_matrices, plot_score_distributions,
                           visualize_heatmaps, browse_gallery,
                           gallery_original_images, gallery_processed_images,
                           gallery_preprocessed_images, gallery_preprocessed_overlay_images)

console = Console()


def main():
    CFG = Config()

    # Fail fast if train.py has not produced the artifacts yet.
    io_utils.require_artifacts(CFG)

    # ── Load everything from disk ─────────────────────────────────────────────
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

    # ── Metrics from saved scores (no model inference) ────────────────────────
    metrics = {split: compute_metrics(d['scores'], d['y_true'], threshold)
               for split, d in data.items()}

    # ── EDA: class distribution per split ─────────────────────────────────────
    plot_class_distribution({
        'Train'      : data['train']['labels'],
        'Validation' : data['val']['labels'],
        'Test'       : data['test']['labels'],
    }, CFG)
    print('EDA chart saved.')

    # ── Training history ──────────────────────────────────────────────────────
    save_file_path = plot_training_history(history, CFG)
    print(f'Training history plot saved successfully at: {save_file_path}')

    # ── ROC / PR / confusion matrices / score distributions ───────────────────
    split_meta = [
        ('Training Set',   metrics['train'], '#2196F3'),
        ('Validation Set', metrics['val'],   '#4CAF50'),
        ('Test Set',       metrics['test'],  '#FF5722'),
    ]
    plot_roc_curves(split_meta, CFG)
    plot_pr_curves(split_meta, CFG)
    plot_confusion_matrices(split_meta, CFG)
    plot_score_distributions(split_meta, threshold, CFG)

    # ── Heatmaps per split ────────────────────────────────────────────────────
    # แบบที่ 1: base image = RGB จริง (แสดงเสมอ)
    # แบบที่ 2: base image = ภาพหลัง preprocessing จริง (grayscale/equalized) —
    #           แสดงเพิ่มเฉพาะเมื่อ COLOR_MODE ไม่ใช่ RGB
    for split, split_name in [('train','Train'), ('val','Validation'), ('test','Test')]:
        d = data[split]
        visualize_heatmaps(
            d['paths'], d['orig_imgs'], d['heatmaps'], d['labels'],
            d['scores'], threshold, split_name, CFG, n_samples=20, image_kind='rgb')

        if CFG.COLOR_MODE != 'RGB':
            visualize_heatmaps(
                d['paths'], d['preproc_imgs'], d['heatmaps'], d['labels'],
                d['scores'], threshold, split_name, CFG, n_samples=20, image_kind='preproc')

    # ── Result gallery ────────────────────────────────────────────────────────
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
    df_gallery.to_csv(f'{CFG.OUTPUT_PATH}/gallery_index.csv', index=False)

    print(f'df_gallery: {len(df_gallery):,} แถว '
          f"(train={len(data['train']['paths']):,}, "
          f"val={len(data['val']['paths']):,}, "
          f"test={len(data['test']['paths']):,})")

    print('\nจำนวนภาพต่อ split × label (ground truth):')
    print(df_gallery.groupby(['split', 'label_gt']).size().unstack(fill_value=0))

    print('\nจำนวนภาพต่อ split × ผลทาย (ถูก/ผิด):')
    print(df_gallery.groupby(['split', 'correct']).size().unstack(fill_value=0))

    # 3-column gallery (original | error map | overlay) — misclassified samples
    _ = browse_gallery(df_gallery, split_arrays, CFG, split='train', correct=False, n=100)
    _ = browse_gallery(df_gallery, split_arrays, CFG, split='test',  correct=False, n=100)
    _ = browse_gallery(df_gallery, split_arrays, CFG, split='val',   correct=False, n=100)

    # ── Output gallery แบบที่ 1: ภาพจริง (RGB, always) ──────────────────────
    # ── Output gallery แบบที่ 2: ภาพหลัง image processing (heatmap overlay on RGB) ─
    for split_name in ['train', 'val', 'test']:
        _ = gallery_original_images(
            df_gallery, split_arrays, CFG,
            split=split_name, n=20, ncols=5)
        _ = gallery_processed_images(
            df_gallery, split_arrays, CFG,
            split=split_name, n=20, ncols=5)

    # ── ถ้า color mode ไม่ใช่ RGB (grayscale / grayscale+equalization) ─────────
    # เพิ่ม gallery อีก 2 แบบที่แสดงภาพซึ่งผ่าน preprocessing จริงที่ป้อนเข้าโมเดล:
    #   - gallery_preprocessed_images         : ภาพ preprocessed ล้วน ๆ (ไม่มี overlay)
    #   - gallery_preprocessed_overlay_images : ภาพ preprocessed + heatmap overlay
    if CFG.COLOR_MODE != 'RGB':
        print(f"\nCOLOR_MODE = '{CFG.COLOR_MODE}' → เพิ่ม gallery เวอร์ชัน preprocessed ด้วย")
        for split_name in ['train', 'val', 'test']:
            _ = gallery_preprocessed_images(
                df_gallery, split_arrays, CFG,
                split=split_name, n=20, ncols=5)
            _ = gallery_preprocessed_overlay_images(
                df_gallery, split_arrays, CFG,
                split=split_name, n=20, ncols=5)

    logger.info('All plots/heatmaps rendered.')


if __name__ == '__main__':
    main()
