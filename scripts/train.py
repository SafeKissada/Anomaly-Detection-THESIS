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


def main():
    logger.info(f'Logging to {_log_file}')


    # ── PHASE 1 — Setup & configuration ──────────────────────────────────────
    CFG = Config()
    set_seed(CFG.SEED)

    # NOTE (fix 2.NEW-1/2.NEW-2/2.NEW-3): the two Panel() calls below used to
    # have COMMAS between what should have been one concatenated multi-line
    # f-string. Python then parsed each line as a SEPARATE positional argument
    # to Panel(), but rich.Panel.__init__() only accepts 2-3 positional args
    # (renderable, box) — so this raised
    #   TypeError: Panel.__init__() takes from 2 to 3 positional arguments
    #   but 6 positional arguments (and 1 keyword-only argument) were given
    # immediately, before main() could do anything else. Fixed by joining
    # every line into a single string via adjacent string-literal
    # concatenation (no comma) ending each line with '\n'. Also fixed two
    # latent bugs that were hiding behind that crash and would have fired
    # next: mismatched/misspelled rich markup tags ('cran'/'cray' instead of
    # 'cyan', and an opening tag that didn't match its closing tag — both
    # raise rich.errors.MarkupError), and CFG.AE_EPOCH (missing the trailing
    # 'S') which does not exist on Config and would have raised
    # AttributeError: 'Config' object has no attribute 'AE_EPOCH'.
    console.print(Panel(
        f'Device        : [bold cyan]{CFG.DEVICE}[/bold cyan]\n'
        f'Backbone      : [bold cyan]ConvNeXt-{CFG.BACKBONE.capitalize()}[/bold cyan]\n'
        f'Color mode    : [bold cyan]{CFG.COLOR_MODE}[/bold cyan]\n'
        f'Loss function : [bold cyan]{CFG.LOSS}[/bold cyan]\n'
        f'Optimizer     : [bold cyan]{CFG.OPTIM}[/bold cyan]\n'
        f'Score method  : [bold cyan]{CFG.SCORE_METHOD}[/bold cyan]\n'
        f'AE monitor    : [bold cyan]{CFG.SCORE_TOPK_PERCENT}[/bold cyan]',
        title='[bold]BACKBONE[/bold]'
    ))

    print('\n')

    console.print(Panel(
        f'SEED          : [bold white]{CFG.SEED}[/bold white]\n'
        f'IMAGE         : [bold white]{CFG.IMAGE_SIZE}[/bold white]\n'
        f'BRIGHTNESS    : [bold white]{CFG.AUG_COLOR_JITTER }[/bold white]\n'
        f'MSE WEIGHT    : [bold white]{CFG.MSE_WEIGHT}[/bold white]\n'
        f'SSIM WEIGHT   : [bold white]{CFG.SSIM_WEIGHT}[/bold white]\n'
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
    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    # Previously this block re-implemented (scan + transforms + dataset +
    # loader, for train/val/test/normal-only) inline, duplicating the exact
    # same logic already available in build_datasets_and_loaders(). Any future
    # change to that logic would have needed to be made in two places at once
    # and could silently drift apart. Now there is a single source of truth.
    io_ = build_datasets_and_loaders(CFG)
    df_train, df_val, df_test = io_['df_train'], io_['df_val'], io_['df_test']
    val_loader   = io_['val_loader']
    test_loader  = io_['test_loader']
    normal_loader = io_['normal_loader']
    # NOTE: build_datasets_and_loaders() still returns a full train_loader/
    # train_ds (good+defect) internally, but train.py deliberately never
    # touches them below — training only ever sees normal_loader, and
    # scoring/reporting only ever runs on val/test. This keeps "train" doing
    # exactly one thing: training the autoencoder on normal images.
    val_ds, test_ds, normal_ds = (
        io_['val_ds'], io_['test_ds'], io_['normal_ds'])

    for name, df in [('TRAIN',df_train),('VAL',df_val),('TEST',df_test)]:
        console.print(f'[{name:5}] total={len(df):,}  {df["label"].value_counts().to_dict()}')

    # Report how many files were excluded per split due to ambiguous/missing
    # filename keywords (previously only visible in the log file, not in any
    # artifact — see .attrs populated by scan_and_split()).
    dropped_counts = {
        name: df.attrs.get('n_dropped_ambiguous_or_unlabelled', 0)
        for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]
    }
    if any(dropped_counts.values()):
        console.print(f'[yellow]Dropped (ambiguous/unlabelled) per split: {dropped_counts}[/yellow]')

    print(f'Train (good+defect, on disk) : {len(df_train):,}  |  '
          f'Normal-only (actually used to train AE) : {len(normal_ds):,}  '
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

    # Report the ACTUAL bottleneck spatial resolution instead of a hardcoded
    # "@4x4" assumption (which silently becomes wrong the moment IMAGE_SIZE
    # or BACKBONE changes). This is also a lightweight, non-invasive way to
    # surface the "how coarse is the heatmap before upsampling?" limitation
    # (see Findings 2.7) at run time rather than only in a static report.
    with torch.no_grad():
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
    # Deliberately NOT scoring the train split here: train.py's only job is to
    # train the autoencoder on normal images; all reported metrics/artifacts
    # come from val (threshold selection) and test (final report) only.
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

    # ── PHASE 8 — Save artifacts & final summary ──────────────────────────────
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

    # Use the epoch actually selected by EarlyStopping (per cfg.AE_MONITOR),
    # not the global min over every epoch ever seen — those are only
    # guaranteed to be the same epoch when AE_MONITOR == 'val_loss'. With the
    # default AE_MONITOR='val_auroc', the previous `min(history['val_loss'])`
    # could silently report a val_loss value that has nothing to do with the
    # checkpoint that was actually saved and loaded.
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
        'results': {
            split: {k: float(v) for k,v in m.items()
                    if isinstance(v, float) and not np.isnan(v)}
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
        f'  Method         : Feature-space Convolutional Autoencoder',
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