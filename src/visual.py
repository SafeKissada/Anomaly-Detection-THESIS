"""Plot functions only — no training, no fitting, no metric computation.
All inputs are precomputed (history dict, metrics dicts, scores, heatmaps)."""

import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_class_distribution(split_labels: Dict[str, List[str]], cfg) -> str:
    """split_labels: {'Train': [...labels...], 'Validation': [...], 'Test': [...]}"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = {'normal':'#4C8BE0','anomaly':'#E05C4C'}
    for ax, (split_name, labels) in zip(axes, split_labels.items()):
        counts = pd.Series(labels).value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[palette.get(l,'grey') for l in counts.index],
                      alpha=0.85, edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_title(f'{split_name}  (n={len(labels):,})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle('Class Distribution per Split', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/eda_class_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out


def plot_training_history(history: Dict, cfg) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    _monitor = cfg.AE_MONITOR
    if _monitor == 'val_auroc':
        best_ep = int(np.argmax(history['val_auroc']))
    elif _monitor == 'val_loss_normal':
        best_ep = int(np.argmin(history['val_loss_normal']))
    else:
        best_ep = int(np.argmin(history['val_loss']))

    # ── 1. Loss curves ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss (normal only)', color='#2196F3', lw=2)
    ax.plot(history['val_loss'],   label='Val Loss  (mixed normal+anomaly)', color='#FF5722', lw=2, linestyle='--')
    ax.axvline(best_ep, color='black', linestyle=':', alpha=0.6,
               label=f'Best epoch {best_ep+1}  (monitor={_monitor})\n'
                     f'(val_loss={history["val_loss"][best_ep]:.6f})')
    ax.set_title('Reconstruction Loss (mixed val)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({cfg.LOSS})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # ── 2. Normal-only validation loss ──────────────────────────────────────
    ax = axes[1]
    ax.plot(history['val_loss_normal'], label='Val Loss (normal-only)', color='#FF9800', lw=2)
    ax.axvline(best_ep, color='black', linestyle=':', alpha=0.6,
               label=f'Best epoch {best_ep+1}\n(val_loss_normal={history["val_loss_normal"][best_ep]:.6f})')
    ax.set_title('Reconstruction Loss (normal-only val)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({cfg.LOSS})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # ── 3. Validation AUROC ──────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(history['val_auroc'], label='Val AUROC', color='#4CAF50', lw=2)
    best_auroc_at_best_ep = history['val_auroc'][best_ep]
    ax.axvline(best_ep, color='black', linestyle=':', alpha=0.6,
               label=f'At Best Epoch:\nAUROC={best_auroc_at_best_ep:.4f}')
    ax.set_title('Validation AUROC Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # ── 4. Learning rate ─────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(history['lr'], color='#9C27B0', lw=2)
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    backbone_name = getattr(cfg, 'BACKBONE', 'tiny')
    plt.suptitle(f'ConvNeXt-{backbone_name.capitalize()} (Stages 2+3) + CAE — Training History',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
    save_file_path = os.path.join(cfg.OUTPUT_PATH, 'training_history.png')
    plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
    plt.show()
    return save_file_path


def plot_roc_curves(split_meta, cfg) -> str:
    """split_meta: list of (name, metrics_dict, color)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, m, color) in zip(axes, split_meta):
        ax.plot(m['fpr'], m['tpr'], color=color, lw=2.5,
                label=f'ROC (AUC = {m["auc"]:.4f})')
        ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
        ax.fill_between(m['fpr'], m['tpr'], alpha=0.08, color=color)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate',  fontsize=11)
        ax.set_title(f'ROC Curve — {name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle(f'ConvNeXt-{cfg.BACKBONE.capitalize()} AE — ROC Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/roc_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out


def plot_pr_curves(split_meta, cfg) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, m, color) in zip(axes, split_meta):
        prec_c, rec_c, _ = precision_recall_curve(m['gt'], m['scores'])
        ax.plot(rec_c, prec_c, color=color, lw=2.5, label=f'AP = {m["ap"]:.4f}')
        ax.fill_between(rec_c, prec_c, alpha=0.08, color=color)
        ax.set_xlabel('Recall', fontsize=11); ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'PR Curve — {name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle(f'ConvNeXt-{cfg.BACKBONE.capitalize()} AE — Precision-Recall Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/pr_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out


def plot_confusion_matrices(split_meta, cfg) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, m, color) in zip(axes, split_meta):
        sns.heatmap(m['cm'], annot=True, fmt='d', ax=ax, cmap='Blues',
                    linewidths=0.5, linecolor='grey',
                    xticklabels=['Normal','Anomaly'],
                    yticklabels=['Normal','Anomaly'],
                    annot_kws={'size':13, 'weight':'bold'}, cbar=False)
        ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'Confusion Matrix — {name}\n'
                     f'Acc={m["acc"]:.3f}  F1={m["f1"]:.3f}',
                     fontsize=12, fontweight='bold')

    plt.suptitle(f'ConvNeXt-{cfg.BACKBONE.capitalize()} AE — Confusion Matrices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/confusion_matrices.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out


def plot_score_distributions(split_meta, threshold: float, cfg) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, m, color) in zip(axes, split_meta):
        s_n = m['scores'][m['gt']==0]
        s_a = m['scores'][m['gt']==1]
        ax.hist(s_n, bins=40, alpha=0.6, color='#2196F3',
                label=f'Normal  (n={len(s_n)})', density=True)
        ax.hist(s_a, bins=40, alpha=0.6, color='#E05C4C',
                label=f'Anomaly (n={len(s_a)})', density=True)
        ax.axvline(threshold, color='black', linestyle='--', lw=2,
                   label=f'Threshold={threshold:.4f}')
        ax.set_xlabel(f'Anomaly Score ({cfg.SCORE_METHOD})', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Score Distribution — {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle(f'ConvNeXt-{cfg.BACKBONE.capitalize()} AE — Score Distributions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/score_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    return out


def overlay_heatmap(image_np: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend jet-coloured heatmap onto [H,W,3] float [0,1] image."""
    heat_8 = (heat.clip(0,1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_8, cv2.COLORMAP_JET)
    heat_rgb   = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    img_u8 = (image_np.clip(0,1) * 255).astype(np.uint8)
    return cv2.addWeighted(img_u8, 1-alpha, heat_rgb, alpha, 0)


def visualize_heatmaps(
    paths     : List[str],
    orig_imgs : List[np.ndarray],
    heatmaps  : List[np.ndarray],
    labels    : List[str],
    scores    : np.ndarray,
    threshold : float,
    split_name: str,
    cfg,
    n_samples : int = 8,
    seed      : int = 42
) -> None:
    """
    แสดง Original | Reconstruction Error Map | Heatmap Overlay ต่อภาพ พร้อม
    ระบุชัดเจนต่อภาพว่า:
      - GT   (Ground Truth) : label จริงของภาพ (normal / anomaly)
      - Pred (Prediction)   : โมเดลทำนายว่าอะไร (จาก score เทียบกับ threshold)
      - ผลถูก/ผิด          : กรอบสีเขียว = ทำนายถูก, กรอบสีแดง = ทำนายผิด
    """
    rng = np.random.default_rng(seed)
    y = np.array([1 if l=='anomaly' else 0 for l in labels])
    idx_n = np.where(y==0)[0]; idx_a = np.where(y==1)[0]
    n_each = n_samples // 2
    sel_n  = rng.choice(idx_n, size=min(n_each, len(idx_n)),  replace=False)
    sel_a  = rng.choice(idx_a, size=min(n_each, len(idx_a)),  replace=False)
    selected = list(sel_n) + list(sel_a)

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.6*n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for col_title, ax in zip(['Original Image','Reconstruction Error Map','Heatmap Overlay'],
                              axes[0]):
        ax.set_title(col_title, fontsize=10, fontweight='bold', pad=45)

    for ri, idx in enumerate(selected):
        orig    = orig_imgs[idx]           # [H,W,3] float [0,1]
        heat    = heatmaps[idx]            # [H,W]   float [0,1]
        overlay = overlay_heatmap(orig, heat, alpha=0.5)
        score   = float(scores[idx])
        gt      = labels[idx]
        pred    = 'anomaly' if score >= threshold else 'normal'
        ok      = (pred == gt)
        clr     = '#1B5E20' if ok else '#B71C1C'
        row_lbl = (f'Actual = {gt.upper()}   |   '
                   f'Predicted = {pred.upper()}   |   '
                   f'Score={score:.4f}   {"✓ True" if ok else "✗ False"}')

        axes[ri,0].imshow(orig); axes[ri,0].axis('off')
        axes[ri,1].imshow(heat, cmap='jet', vmin=0, vmax=1); axes[ri,1].axis('off')
        axes[ri,2].imshow(overlay); axes[ri,2].axis('off')

        axes[ri,1].set_title(
            row_lbl, fontsize=9, fontweight='bold', color=clr, pad=10,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=clr,
                       boxstyle='round,pad=0.35', linewidth=1.5)
        )

        for ax in axes[ri]:
            for sp in ax.spines.values():
                sp.set_edgecolor(clr); sp.set_linewidth(2.5)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        f'Heatmap Visualisation — {split_name} | ConvNeXt-{cfg.BACKBONE.capitalize()} AE',
        fontsize=13, fontweight='bold', y=1.005
    )
    plt.subplots_adjust(hspace=0.55, wspace=0.08)
    plt.tight_layout()
    out = f'{cfg.OUTPUT_PATH}/heatmaps_{split_name.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Heatmaps saved → {out}')


def _get_display_image(split_arrays: Dict[str, Dict], split: str, idx: int, mode: str) -> np.ndarray:
    """mode='original' -> raw image as-is.
    mode='processed'  -> raw image after image processing (heatmap overlay)."""
    arrs = split_arrays[split]
    orig = arrs['imgs'][idx]
    if mode == 'original':
        return orig
    elif mode == 'processed':
        heat = arrs['hmaps'][idx]
        return overlay_heatmap(orig, heat, alpha=0.5)
    else:
        raise ValueError(f"mode must be 'original' or 'processed', got {mode!r}")


def render_image_gallery(
    df_gallery  : pd.DataFrame,
    split_arrays: Dict[str, Dict],
    cfg,
    mode        : str  = 'original',   # 'original' | 'processed'
    query       : str  = None,
    split       : str  = None,
    label       : str  = None,
    pred_label  : str  = None,
    correct     : bool = None,
    n           : int  = 20,
    ncols       : int  = 5,
    random_state: int  = 42,
) -> pd.DataFrame:
    """Contact-sheet style gallery showing ONE kind of image per cell
    (either the untouched original photo, or the image after processing —
    i.e. heatmap overlay). Each thumbnail is captioned with filename,
    ground-truth, prediction, and score; border colour = green(correct)/red(wrong)."""
    assert mode in ('original', 'processed'), \
        "mode must be 'original' or 'processed'"

    df = df_gallery.copy()
    if split is not None:
        df = df[df['split'] == split]
    if label is not None:
        df = df[df['label_gt'] == label]
    if pred_label is not None:
        df = df[df['pred_label'] == pred_label]
    if correct is not None:
        df = df[df['correct'] == correct]
    if query is not None:
        df = df[df['filename'].str.contains(query, case=False, na=False)]

    if len(df) == 0:
        print('ไม่พบภาพที่ตรงเงื่อนไขที่กรอง')
        return df

    if len(df) > n:
        df = df.sample(n=n, random_state=random_state)
    df = df.sort_values(['split', 'idx_in_split']).reset_index(drop=True)

    n_imgs = len(df)
    ncols  = min(ncols, n_imgs)
    nrows  = int(np.ceil(n_imgs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for ax in axes.flat:
        ax.axis('off')

    for i, row in df.iterrows():
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img = _get_display_image(split_arrays, row['split'], row['idx_in_split'], mode)
        ax.imshow(img, cmap=None if mode == 'processed' else None)

        ok  = row['correct']
        clr = '#1B5E20' if ok else '#B71C1C'
        cap = (f"{row['filename'][:20]}\n"
               f"GT={row['label_gt']}  Pred={row['pred_label']}\n"
               f"Score={row['score']:.3f}  {'✓' if ok else '✗'}")
        ax.set_title(cap, fontsize=7.5, fontweight='bold', color=clr, pad=6)

        ax.axis('on')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(clr); sp.set_linewidth(2.2)

    # หมายเหตุ: ใช้ชื่อภาษาอังกฤษบนกราฟ เพราะฟอนต์ default ของ matplotlib ส่วนใหญ่
    # ไม่มี glyph ภาษาไทย (จะขึ้น warning/กล่องสี่เหลี่ยมแทนตัวอักษร)
    mode_title = 'Original Images' if mode == 'original' \
        else 'Processed Images (Heatmap Overlay)'
    backbone_name = getattr(cfg, 'BACKBONE', 'tiny')
    plt.suptitle(f'{mode_title}\nConvNeXt-{backbone_name.capitalize()} AE Gallery',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    name_parts = [p for p in [split, label, pred_label] if p]
    if correct is not None:
        name_parts.append('correct' if correct else 'incorrect')
    suffix = '_'.join(name_parts) or 'all'
    out = f'{cfg.OUTPUT_PATH}/gallery_{mode}_{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Gallery [{mode}] saved → {out}')
    plt.show()
    return df


def gallery_original_images(df_gallery, split_arrays, cfg, **kwargs) -> pd.DataFrame:
    """Gallery แบบที่ 1: แสดงเฉพาะภาพจริง (ก่อนผ่าน image processing)."""
    return render_image_gallery(df_gallery, split_arrays, cfg, mode='original', **kwargs)


def gallery_processed_images(df_gallery, split_arrays, cfg, **kwargs) -> pd.DataFrame:
    """Gallery แบบที่ 2: แสดงภาพหลังผ่าน image processing (heatmap overlay)."""
    return render_image_gallery(df_gallery, split_arrays, cfg, mode='processed', **kwargs)


def browse_gallery(
    df_gallery  : pd.DataFrame,
    split_arrays: Dict[str, Dict],
    cfg        = None,
    query      : str  = None,
    split      : str  = None,
    label      : str  = None,
    pred_label : str  = None,
    correct    : bool = None,
    n          : int  = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df_gallery.copy()
    if split is not None:
        df = df[df['split'] == split]
    if label is not None:
        df = df[df['label_gt'] == label]
    if pred_label is not None:
        df = df[df['pred_label'] == pred_label]
    if correct is not None:
        df = df[df['correct'] == correct]
    if query is not None:
        df = df[df['filename'].str.contains(query, case=False, na=False)]

    if len(df) == 0:
        print('ไม่พบภาพที่ตรงเงื่อนไขที่กรอง')
        return df

    if len(df) > n:
        df = df.sample(n=n, random_state=random_state)
    df = df.sort_values(['split', 'idx_in_split']).reset_index(drop=True)

    n_rows = len(df)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_title, ax in zip(['Original Image', 'Reconstruction Error Map', 'Heatmap Overlay'],
                              axes[0]):
        ax.set_title(col_title, fontsize=10, fontweight='bold', pad=45)

    for ri, row in df.iterrows():
        arrs = split_arrays[row['split']]
        i    = row['idx_in_split']
        orig = arrs['imgs'][i]
        heat = arrs['hmaps'][i]
        overlay = overlay_heatmap(orig, heat, alpha=0.5)

        ok  = row['correct']
        clr = '#1B5E20' if ok else '#B71C1C'
        row_lbl = (
            f"[{row['split'].upper()}] {row['filename']}\n"
            f"Actual={row['label_gt'].upper()}   "
            f"Predicted={row['pred_label'].upper()}   "
            f"Score={row['score']:.4f}   {'✓ True' if ok else '✗ False'}"
        )

        axes[ri, 0].imshow(orig); axes[ri, 0].axis('off')
        axes[ri, 1].imshow(heat, cmap='jet', vmin=0, vmax=1); axes[ri, 1].axis('off')
        axes[ri, 2].imshow(overlay); axes[ri, 2].axis('off')

        axes[ri, 1].set_title(
            row_lbl, fontsize=8.5, fontweight='bold', color=clr, pad=10,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=clr,
                       boxstyle='round,pad=0.35', linewidth=1.5)
        )

        for ax in axes[ri]:
            for sp in ax.spines.values():
                sp.set_edgecolor(clr); sp.set_linewidth(2.5)
            ax.set_xticks([]); ax.set_yticks([])

    plt.subplots_adjust(hspace=0.6, wspace=0.08)
    plt.tight_layout()
    if cfg is not None:
        name_parts = [p for p in [split, label, pred_label] if p]
        if correct is not None:
            name_parts.append('correct' if correct else 'incorrect')
        out = f"{cfg.OUTPUT_PATH}/gallery_{'_'.join(name_parts) or 'all'}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Gallery saved → {out}')
    plt.show()
    return df
