"""จุดเริ่มต้นการเทรน: setup → train → score → save ทุก artifact ผ่าน io_utils

ตัด EDA/visualization ออกโดยตั้งใจ — รัน scripts/visualize.py ทีหลังเพื่อ
render กราฟ/heatmap ทั้งหมดจาก artifact ที่เซฟไว้ในนี้

Training entry point: setup → train → score → save all artifacts via io_utils.

EDA/visualization is intentionally excluded — run scripts/visualize.py afterwards
to render all plots/heatmaps from the artifacts saved here.
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging: console + logs/train_{timestamp}.log ────────────────────────────
# บันทึก log ทั้งขึ้นหน้าจอ (console) และลงไฟล์พร้อมกัน
# Log to both the console and a timestamped file simultaneously.
_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_dir = PROJECT_ROOT / 'logs'
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / f'train_{_timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding='utf-8'),
    ])
logger = logging.getLogger('ConvNeXtAutoencoder')


import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.model.backbone_baseline import ConvNeXtExtractor
from src.model.autoencoder import FeatureAutoencoder
from src.engine import train_autoencoder, score_dataset_split, get_best_epoch
from src.evaluate import (compute_metrics, select_percentile_threshold,
                          oracle_threshold_diagnostic)
from src import io_utils

console = Console()


def make_pred_df(paths, labels, scores, metrics):
    df = pd.DataFrame({
        'path':    paths,
        'label':   labels,
        'y_true':  metrics['gt'].tolist(),
        'score':   scores.tolist(),
        'y_pred':  metrics['pred'].tolist(),
        'correct': (metrics['gt'] == metrics['pred']).astype(int).tolist(),
    })
    return df


def main(cfg: Config = None):
    """รัน pipeline เต็มรูปแบบ: train -> score -> save

    Args:
      cfg: instance ของ Config ที่สร้างไว้แล้วสำหรับ run นี้ (เช่น สร้างจาก
        CLI arguments โดย scripts/run_train.py) ถ้าไม่ส่งมา (ค่า default)
        จะใช้ Config() ตามค่าปัจจุบันใน config/config.py ทำให้พฤติกรรมของ
        `python scripts/train.py` แบบเดิมเหมือนเดิมทุกประการ

    Run the full train -> score -> save pipeline.

    Args:
      cfg: an already-constructed Config instance to use for this run (e.g.
        built from CLI arguments by scripts/run_train.py). If omitted
        (the default), a Config() with the values currently in
        config/config.py is used, preserving the original
        `python scripts/train.py` behavior exactly.
    """
    logger.info(f'Logging to {_log_file}')


    # ── PHASE 1 — Setup & configuration / ตั้งค่าเริ่มต้น ──────────────────
    CFG = cfg if cfg is not None else Config()
    set_seed(CFG.SEED)

    # ต่อท้ายรายละเอียด optimizer เฉพาะตัวเข้า panel (SGD/RMSprop มี
    # hyperparameter เพิ่มที่ Adam/AdamW ไม่มี)
    # Append optimizer-specific detail to the panel (SGD/RMSprop have
    # extra hyperparameters that Adam/AdamW don't).
    _optim_upper = CFG.OPTIM.strip().upper()
    if _optim_upper == 'SGD':
        _optim_detail = (f'  (momentum={CFG.AE_MOMENTUM}, '
                         f'nesterov={CFG.AE_SGD_NESTEROV and CFG.AE_MOMENTUM > 0})')
    elif _optim_upper == 'RMSPROP':
        _optim_detail = (f'  (momentum={CFG.AE_MOMENTUM}, '
                         f'alpha={CFG.AE_RMSPROP_ALPHA}, eps={CFG.AE_RMSPROP_EPS})')
    else:
        _optim_detail = ''

    console.print(Panel(
        f'Device        : [bold cyan]{CFG.DEVICE}[/bold cyan]\n'
        f'Backbone      : [bold cyan]ConvNeXt-{CFG.BACKBONE.capitalize()}[/bold cyan]\n'
        f'Color mode    : [bold cyan]{CFG.COLOR_MODE}[/bold cyan]\n'
        f'Loss function : [bold cyan]{CFG.LOSS}[/bold cyan]\n'
        f'Optimizer     : [bold cyan]{CFG.OPTIM}[/bold cyan][cyan]{_optim_detail}[/cyan]\n'
        f'Score method  : [bold cyan]{CFG.SCORE_METHOD}[/bold cyan]\n'
        f'AE monitor    : [bold cyan]{CFG.SCORE_TOPK_PERCENT}[/bold cyan]',
        title='[bold]BACKBONE[/bold]'
    ))

    print('\n')

    console.print(Panel(
        f'SEED          : [bold white]{CFG.SEED}[/bold white]\n'
        f'IMAGE         : [bold white]{CFG.IMAGE_SIZE}[/bold white]\n'
        f'BRIGHTNESS    : [bold white]{CFG.AUG_COLOR_JITTER }[/bold white]\n'
        f'HUBER DELTA   : [bold white]{CFG.HUBER_DELTA}[/bold white]\n'
        f'COS LAM       : [bold white]{CFG.COS_LAM}[/bold white]\n'
        f'BATCH SIZE    : [bold white]{CFG.BATCH_SIZE}[/bold white]\n'
        f'EPOCHS        : [bold white]{CFG.AE_EPOCHS}[/bold white]\n'
        f'LR            : [bold white]{CFG.AE_LR}[/bold white]\n'
        f'WEIGHT DECAY  : [bold white]{CFG.AE_WEIGHT_DECAY}[/bold white]\n'
        f'BOTTLENECK    : [bold white]{CFG.AE_BOTTLENECK_CH} CH[/bold white]\n'
        f'PATIENCE      : [bold white]{CFG.AE_PATIENCE}[/bold white]\n'
        f'STEP          : [bold white]{CFG.AE_LR_STEP}[/bold white]\n'
        f'GAMMA         : [bold white]{CFG.AE_LR_GAMMA}[/bold white]\n'
        f'HEATMAP SIGMA : [bold red]{CFG.HEATMAP_SIGMA}[/bold red]\n'
        f'THRESHOLD     : [bold red]{CFG.THRESHOLD_PERCENTILE}%[/bold red]\n'
        f'TOP-K%        : [bold red]{CFG.SCORE_TOPK_PERCENT}%[/bold red]',
        title='[bold]Parameter[/bold]'
    ))
    # ── Datasets & DataLoaders / สร้างชุดข้อมูลและตัวโหลด ──────────────────
    io_ = build_datasets_and_loaders(CFG)
    df_train, df_val, df_test = io_['df_train'], io_['df_val'], io_['df_test']
    val_loader   = io_['val_loader']
    test_loader  = io_['test_loader']
    normal_loader = io_['normal_loader']
    val_ds, test_ds, normal_ds = (
        io_['val_ds'], io_['test_ds'], io_['normal_ds'])

    for name, df in [('TRAIN',df_train),('VAL',df_val),('TEST',df_test)]:
        console.print(f'[{name:5}] total={len(df):,}  {df["label"].value_counts().to_dict()}')

    dropped_counts = {
        name: df.attrs.get('n_dropped_ambiguous_or_unlabelled', 0)
        for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]
    }
    if any(dropped_counts.values()):
        console.print(f'[yellow]Dropped (ambiguous/unlabelled) per split: {dropped_counts}[/yellow]')

    print(f'Train (good only, used to train AE) : {len(normal_ds):,}  '
          f'(augmentation={"ON" if CFG.USE_AUGMENTATION else "OFF"})')
    print(f'Val   : {len(val_ds):,}')
    print(f'Test  : {len(test_ds):,}')

    # ── PHASE 2 — ConvNeXt feature extractor / ตัวสกัด feature ────────────────
    extractor = ConvNeXtExtractor(variant=CFG.BACKBONE).to(CFG.DEVICE)
    extractor.fit_normalization(normal_loader, CFG.DEVICE)
    io_utils.save_norm_stats(extractor, CFG)
    console.print(Panel(
        f'Feature mean (first 5 ch) : [cyan]{extractor.feat_mean.flatten()[:5].tolist()}[/cyan]\n'
        f'Feature std  (first 5 ch) : [cyan]{extractor.feat_std.flatten()[:5].tolist()}[/cyan]',
        title='[bold]Feature Normalization Fitted (normal-only)[/bold]'
    ))

    # ── PHASE 3 — Autoencoder / สร้าง autoencoder ───────────────────────────────
    ae = FeatureAutoencoder(
        feat_ch=extractor.out_channels,
        bottleneck_ch=CFG.AE_BOTTLENECK_CH
    ).to(CFG.DEVICE)

    total_ae = sum(p.numel() for p in ae.parameters())

    with torch.no_grad():
      # dummy forward ผ่าน bottleneck เพื่อดูขนาด spatial ที่แท้จริง (ไว้เตือน
      # ถ้าเล็กเกินไปจนกระทบความละเอียดของ error map)
      # Dummy forward pass through the bottleneck just to read its actual
      # spatial size (used to warn if it's too small for the error map).
      _dummy_feat = torch.zeros(1, extractor.out_channels, *extractor.spatial_size).to(CFG.DEVICE)
      _bneck_shape = tuple(ae.bottleneck(_dummy_feat).shape[-2:])
    bneck_note = ''
    if min(_bneck_shape) <= 4:
      bneck_note = (
          f'  [yellow]note: bottleneck is only {_bneck_shape[0]}×{_bneck_shape[1]} — '
          f'error maps at this resolution are coarse before upsampling; small/'
          f'thin defects may be smoothed away by the Gaussian blur step[/yellow]\n'
      )
    console.print(Panel(
        f'Feature channels : [cyan]{extractor.out_channels}[/cyan]\n'
        f'Bottleneck       : [cyan]{CFG.AE_BOTTLENECK_CH} ch @ '
        f'{_bneck_shape[0]}×{_bneck_shape[1]}[/cyan]\n'
        f'{bneck_note}'
        f'AE params        : [cyan]{total_ae:,}[/cyan]  (all trainable)',
        title='[bold]Autoencoder Ready[/bold]'
    ))

    # ── PHASE 4 — Autoencoder training / เทรน autoencoder ──────────────────────
    print(f'Training on {len(normal_loader.dataset):,} normal images  |  '
          f'Val on {len(val_loader.dataset):,} images')
    print(f'Device: {CFG.DEVICE}  |  Epochs: {CFG.AE_EPOCHS}  |  Patience: {CFG.AE_PATIENCE}')
    print('─' * 60)

    t_start = time.time()
    history = train_autoencoder(ae, extractor, normal_loader, val_loader, CFG)
    t_end = time.time()
    print(f'\nTotal training time: {(t_end - t_start)/60:.1f} min')
    print('─' * 60)

    io_utils.save_history(history, CFG)

    # ── PHASE 5 — Anomaly scoring / คำนวณ anomaly score ─────────────────────────
    print('=== Scoring val/test splits ===')
    (val_scores, val_y, val_paths, val_labels,
     val_hmaps,  val_imgs, val_preproc_imgs) = score_dataset_split(
        val_loader, extractor, ae, CFG, desc='Score-Val  ')

    (test_scores, test_y, test_paths, test_labels,
     test_hmaps,  test_imgs, test_preproc_imgs) = score_dataset_split(
        test_loader, extractor, ae, CFG, desc='Score-Test ')

    threshold = select_percentile_threshold(val_scores, val_y, CFG)
    print(f'\nDeployment threshold ({CFG.THRESHOLD_PERCENTILE:.0f}th pct of val normal): {threshold:.6f}')

    oracle_threshold, oracle_f1 = oracle_threshold_diagnostic(val_scores, val_y)
    print(f'[Diagnostic/Oracle] Max-F1 threshold on Val (uses val anomaly labels, '
          f'NOT used for reported metrics): {oracle_threshold:.6f}  (F1={oracle_f1:.4f})')
    print(f'[Deployment]         Percentile threshold actually used below       : {threshold:.6f}')

    val_metrics   = compute_metrics(val_scores,   val_y,   threshold)
    test_metrics  = compute_metrics(test_scores,  test_y,  threshold)

    table = Table(title=f'ConvNeXt-{CFG.BACKBONE.capitalize()} Autoencoder — Results',
                  show_header=True, header_style='bold magenta')
    table.add_column('Split',     style='bold')
    table.add_column('AUC-ROC',   justify='right')
    table.add_column('Avg. Prec', justify='right')
    table.add_column('Accuracy',  justify='right')
    table.add_column('Precision', justify='right')
    table.add_column('Recall',    justify='right')
    table.add_column('F1',        justify='right')
    for name, m in [('Validation',val_metrics),('Test',test_metrics)]:
        table.add_row(name,
            f'{m["auc"]:.4f}', f'{m["ap"]:.4f}', f'{m["acc"]:.4f}',
            f'{m["precision"]:.4f}', f'{m["recall"]:.4f}', f'{m["f1"]:.4f}')
    console.print(table)

    # ── PHASE 8 — Save artifacts & final summary / เซฟผลลัพธ์ทั้งหมด ─────────────
    io_utils.save_scores('val', val_scores, val_y, val_paths, val_labels,
                         val_hmaps, val_imgs, CFG, preproc_imgs=val_preproc_imgs)
    io_utils.save_scores('test', test_scores, test_y, test_paths, test_labels,
                         test_hmaps, test_imgs, CFG, preproc_imgs=test_preproc_imgs)
    io_utils.save_threshold(threshold, CFG.THRESHOLD_PERCENTILE,
                            oracle_threshold, oracle_f1, CFG)

    ae_final_path = io_utils.checkpoint_path(CFG, io_utils.FINAL_CKPT_FILE)
    torch.save({'model_state': ae.state_dict(),
                'backbone':    CFG.BACKBONE,
                'feat_ch':     extractor.out_channels,
                'bottleneck':  CFG.AE_BOTTLENECK_CH,
                'threshold':   threshold,
                'timestamp':   datetime.now().isoformat()}, ae_final_path)
    print(f'AE weights saved → {ae_final_path}')

    make_pred_df(val_paths,   val_labels,   val_scores,   val_metrics).to_csv(
        f'{CFG.OUTPUT_PATH}/predictions_val.csv',   index=False)
    make_pred_df(test_paths,  test_labels,  test_scores,  test_metrics).to_csv(
        f'{CFG.OUTPUT_PATH}/predictions_test.csv',  index=False)
    
    best_ep = get_best_epoch(history, CFG.AE_MONITOR)

    summary_dict = {
        'experiment' : CFG.EXPERIMENT,
        'backbone'   : f'ConvNeXt-{CFG.BACKBONE.capitalize()}',
        'method'     : 'Convolutional Autoencoder (feature-space reconstruction)',
        'feat_ch'    : extractor.out_channels,
        'bottleneck' : CFG.AE_BOTTLENECK_CH,
        'threshold'  : float(threshold),
        'ae_epochs'  : len(history['train_loss']),
        'best_epoch' : best_ep + 1,
        'monitor'    : CFG.AE_MONITOR,
        'best_val_loss'  : float(history['val_loss'][best_ep]),
        'best_val_auroc' : float(history['val_auroc'][best_ep]),
        'dropped_ambiguous_or_unlabelled_per_split': dropped_counts,
        # เก็บ snapshot ของทุก field ใน Config ที่ใช้ใน run นี้ (LOSS,
        # HUBER_DELTA, COS_LAM, ทุก hyperparameter AE_*, SCORE_METHOD,
        # SPLIT_RATIOS, SEED, color mode ฯลฯ) — ทำให้ไฟล์ผลลัพธ์นี้อธิบาย
        # ตัวเองได้ครบ (self-documenting) สามารถแยกแยะแต่ละ experiment
        # (เช่น E0 vs E1 vs E2) ได้จาก final_results.json ของมันเองเลย
        # โดยไม่ต้องย้อนไปดูว่า config.py ตอนนั้นตั้งค่าอะไรไว้
        #
        # Full snapshot of every Config field used for this run (LOSS,
        # HUBER_DELTA, COS_LAM, all AE_*
        # hyperparameters, SCORE_METHOD, SPLIT_RATIOS, SEED, color mode,
        # etc.) — makes this results file self-documenting so experiments
        # (e.g. E0 vs E1 vs E2) can be told apart later purely from their
        # own final_results.json, without needing to separately track down
        # which config.py values were in effect when each one was run.
        'config': io_utils.config_to_serializable_dict(CFG),
        'results': {
            split: {
                k: (float(v) if isinstance(v, float) else v)
                for k, v in m.items()
                # เก็บทั้ง float (auc, f1, escape_rate ฯลฯ) และ int
                # (tt/tf/ft/ff — จำนวนนับ confusion matrix ดิบ) กัน array
                # ทิ้ง (cm, gt, pred, fpr, tpr, scores) เพราะ isinstance
                # ไม่ match numpy.ndarray และ NaN (จาก auc/ap ตอนมี class
                # เดียวใน scores) ถูกกรองออกเหมือนเดิม
                #
                # Keep both float (auc, f1, escape_rate, etc.) and int
                # (tt/tf/ft/ff — raw confusion-matrix counts) values;
                # arrays (cm, gt, pred, fpr, tpr, scores) are excluded
                # since isinstance() doesn't match numpy.ndarray, and NaN
                # (from auc/ap when scores contain only one class) is
                # still filtered out as before.
                if (isinstance(v, float) and not np.isnan(v))
                or (isinstance(v, int) and not isinstance(v, bool))
            }
            for split, m in [('val',val_metrics),('test',test_metrics)]
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(f'{CFG.OUTPUT_PATH}/final_results.json','w') as f:
        json.dump(summary_dict, f, indent=2)

    lines = [
        f'Seed                       : {CFG.SEED}',
        f'Autoencoder EPOCHS         : {CFG.AE_EPOCHS}',
        f'Autoencoder Learning Rate  : {CFG.AE_LR}',
        f'Autoencoder weight decay   : {CFG.AE_WEIGHT_DECAY}',
        f'Autoencoder Learning Gamma : {CFG.AE_LR_GAMMA}',
        f'Heatmap Sigma              : {CFG.HEATMAP_SIGMA}',
        f'Threshold Percentile       : {CFG.THRESHOLD_PERCENTILE}',
        '  ── Model   ───────────────────────────────',
        f'[bold]ConvNeXt-{CFG.BACKBONE.capitalize()} Autoencoder — Final Summary[/bold]','',
        '  Method         : Feature-space Convolutional Autoencoder',
        f'  Loss Function  : {CFG.LOSS}',
        f'  Optimization   : {CFG.OPTIM}',
        f'  Feature dim    : {extractor.out_channels} ch (Stage2+Stage3)',
        f'  Bottleneck     : {CFG.AE_BOTTLENECK_CH} ch @ {_bneck_shape[0]}×{_bneck_shape[1]}',
        f'  Trained epochs : {len(history["train_loss"])}  (best epoch={best_ep+1}, '
        f'val_loss={history["val_loss"][best_ep]:.6f}, monitor={CFG.AE_MONITOR})',
        f'  Threshold      : {threshold:.4f}  (val {CFG.THRESHOLD_PERCENTILE:.0f}th pct of normal)','',
        '  ── Test Set Results ─────────────────────',
        f'  AUC-ROC   : {test_metrics["auc"]:.4f}',
        f'  Avg Prec  : {test_metrics["ap"]:.4f}',
        f'  Accuracy  : {test_metrics["acc"]:.4f}',
        f'  Precision : {test_metrics["precision"]:.4f}',
        f'  Recall    : {test_metrics["recall"]:.4f}',
        f'  F1        : {test_metrics["f1"]:.4f}',
    ]
    console.print(Panel('\n'.join(lines),
                  title='[bold green]✓ Experiment Complete[/bold green]', padding=(1,2)))

    print('\nAll output files:')
    for p in sorted(Path(CFG.OUTPUT_PATH).glob('*')):
        print(f'  {p.name}  ({p.stat().st_size/1024:.1f} KB)')

    logger.info('Training pipeline finished. Run scripts/visualize.py to render plots.')


if __name__ == '__main__':
    main()