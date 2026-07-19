"""Training entry point: setup → train → score → save all artifacts via io_utils.

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
from src.data.dataset import (AnomalyDataset, scan_directory, build_transforms,
                              make_loader)
from src.model.backbone_baseline import ConvNeXtExtractor
from src.model.autoencoder import FeatureAutoencoder
from src.engine import train_autoencoder, score_dataset_split
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


def main():
    logger.info(f'Logging to {_log_file}')


    # ── PHASE 1 — Setup & configuration ──────────────────────────────────────
    CFG = Config()
    set_seed(CFG.SEED)

    console.print(Panel(
        f'Device     : [bold cyan]{CFG.DEVICE}[/bold cyan]\n'
        f'Backbone   : [bold cyan]ConvNeXt-{CFG.BACKBONE.capitalize()}[/bold cyan]\n'
        f'AE Epochs  : [bold cyan]{CFG.AE_EPOCHS}[/bold cyan]\n'
        f'LR         : [bold cyan]{CFG.AE_LR}[/bold cyan]\n'
        f'Bottleneck : [bold cyan]{CFG.AE_BOTTLENECK_CH} ch[/bold cyan]\n'
        f'Color mode : [bold cyan]{CFG.COLOR_MODE}[/bold cyan]',
        title='[bold]Config Loaded[/bold]'
    ))

    df_train = scan_directory(CFG.TRAIN_DIR, CFG)
    df_val   = scan_directory(CFG.VAL_DIR,   CFG)
    df_test  = scan_directory(CFG.TEST_DIR,  CFG)

    for name, df in [('TRAIN',df_train),('VAL',df_val),('TEST',df_test)]:
        console.print(f'[{name:5}] total={len(df):,}  {df["label"].value_counts().to_dict()}')

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    imagenet_tf, train_aug_tf, display_tf = build_transforms(CFG)

    train_ds = AnomalyDataset(df_train, imagenet_tf, display_tf, CFG.IMAGE_SIZE)
    val_ds   = AnomalyDataset(df_val,   imagenet_tf, display_tf, CFG.IMAGE_SIZE)
    test_ds  = AnomalyDataset(df_test,  imagenet_tf, display_tf, CFG.IMAGE_SIZE)

    train_loader = make_loader(train_ds, CFG, shuffle=True)
    val_loader   = make_loader(val_ds,   CFG)
    test_loader  = make_loader(test_ds,  CFG)

    df_train_normal = df_train[df_train['label']=='normal'].reset_index(drop=True)
    normal_norm_tf = train_aug_tf if CFG.USE_AUGMENTATION else imagenet_tf
    normal_ds      = AnomalyDataset(df_train_normal, normal_norm_tf, display_tf, CFG.IMAGE_SIZE)
    normal_loader  = make_loader(normal_ds, CFG, shuffle=True)

    print(f'Train : {len(train_ds):,}  |  Normal-only : {len(normal_ds):,}  '
          f'(augmentation={"ON" if CFG.USE_AUGMENTATION else "OFF"})')
    print(f'Val   : {len(val_ds):,}')
    print(f'Test  : {len(test_ds):,}')

    # ── PHASE 2 — ConvNeXt feature extractor ─────────────────────────────────
    extractor = ConvNeXtExtractor(variant=CFG.BACKBONE).to(CFG.DEVICE)
    extractor.fit_normalization(normal_loader, CFG.DEVICE)
    io_utils.save_norm_stats(extractor, CFG)
    console.print(Panel(
        f'Feature mean (first 5 ch) : [cyan]{extractor.feat_mean.flatten()[:5].tolist()}[/cyan]\n'
        f'Feature std  (first 5 ch) : [cyan]{extractor.feat_std.flatten()[:5].tolist()}[/cyan]',
        title='[bold]Feature Normalization Fitted (normal-only)[/bold]'
    ))

    # ── PHASE 3 — Autoencoder ─────────────────────────────────────────────────
    ae = FeatureAutoencoder(
        feat_ch=extractor.out_channels,
        bottleneck_ch=CFG.AE_BOTTLENECK_CH
    ).to(CFG.DEVICE)

    total_ae = sum(p.numel() for p in ae.parameters())
    console.print(Panel(
        f'Feature channels : [cyan]{extractor.out_channels}[/cyan]\n'
        f'Bottleneck       : [cyan]{CFG.AE_BOTTLENECK_CH} ch[/cyan]\n'
        f'AE params        : [cyan]{total_ae:,}[/cyan]  (all trainable)',
        title='[bold]Autoencoder Ready[/bold]'
    ))

    # ── PHASE 4 — Autoencoder training ────────────────────────────────────────
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

    # ── PHASE 5 — Anomaly scoring ─────────────────────────────────────────────
    print('=== Scoring all splits ===')
    (train_scores, train_y, train_paths, train_labels,
     train_hmaps,  train_imgs) = score_dataset_split(
        train_loader, extractor, ae, CFG, desc='Score-Train')

    (val_scores, val_y, val_paths, val_labels,
     val_hmaps,  val_imgs) = score_dataset_split(
        val_loader, extractor, ae, CFG, desc='Score-Val  ')

    (test_scores, test_y, test_paths, test_labels,
     test_hmaps,  test_imgs) = score_dataset_split(
        test_loader, extractor, ae, CFG, desc='Score-Test ')

    threshold = select_percentile_threshold(val_scores, val_y, CFG)
    print(f'\nDeployment threshold ({CFG.THRESHOLD_PERCENTILE:.0f}th pct of val normal): {threshold:.6f}')

    oracle_threshold, oracle_f1 = oracle_threshold_diagnostic(val_scores, val_y)
    print(f'[Diagnostic/Oracle] Max-F1 threshold on Val (uses val anomaly labels, '
          f'NOT used for reported metrics): {oracle_threshold:.6f}  (F1={oracle_f1:.4f})')
    print(f'[Deployment]         Percentile threshold actually used below       : {threshold:.6f}')

    train_metrics = compute_metrics(train_scores, train_y, threshold)
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
    for name, m in [('Train',train_metrics),('Validation',val_metrics),('Test',test_metrics)]:
        table.add_row(name,
            f'{m["auc"]:.4f}', f'{m["ap"]:.4f}', f'{m["acc"]:.4f}',
            f'{m["precision"]:.4f}', f'{m["recall"]:.4f}', f'{m["f1"]:.4f}')
    console.print(table)

    # ── PHASE 8 — Save artifacts & final summary ──────────────────────────────
    io_utils.save_scores('train', train_scores, train_y, train_paths, train_labels,
                         train_hmaps, train_imgs, CFG)
    io_utils.save_scores('val', val_scores, val_y, val_paths, val_labels,
                         val_hmaps, val_imgs, CFG)
    io_utils.save_scores('test', test_scores, test_y, test_paths, test_labels,
                         test_hmaps, test_imgs, CFG)
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

    make_pred_df(train_paths, train_labels, train_scores, train_metrics).to_csv(
        f'{CFG.OUTPUT_PATH}/predictions_train.csv', index=False)
    make_pred_df(val_paths,   val_labels,   val_scores,   val_metrics).to_csv(
        f'{CFG.OUTPUT_PATH}/predictions_val.csv',   index=False)
    make_pred_df(test_paths,  test_labels,  test_scores,  test_metrics).to_csv(
        f'{CFG.OUTPUT_PATH}/predictions_test.csv',  index=False)

    summary_dict = {
        'experiment' : CFG.EXPERIMENT,
        'backbone'   : f'ConvNeXt-{CFG.BACKBONE.capitalize()}',
        'method'     : 'Convolutional Autoencoder (feature-space reconstruction)',
        'feat_ch'    : extractor.out_channels,
        'bottleneck' : CFG.AE_BOTTLENECK_CH,
        'threshold'  : float(threshold),
        'ae_epochs'  : len(history['train_loss']),
        'best_val_loss': float(min(history['val_loss'])),
        'results': {
            split: {k: float(v) for k,v in m.items()
                    if isinstance(v, float) and not np.isnan(v)}
            for split, m in [('train',train_metrics),('val',val_metrics),('test',test_metrics)]
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
        f'  Method         : Feature-space Convolutional Autoencoder',
        f'  Loss Function  : {CFG.LOSS}',
        f'  Optimization   : {CFG.OPTIM}',
        f'  Feature dim    : {extractor.out_channels} ch (Stage2+Stage3)',
        f'  Bottleneck     : {CFG.AE_BOTTLENECK_CH} ch @ 4×4',
        f'  Trained epochs : {len(history["train_loss"])}  (best val={min(history["val_loss"]):.6f})',
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
