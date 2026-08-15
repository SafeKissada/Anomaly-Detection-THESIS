"""Metric ต่างๆ, percentile threshold, และ oracle threshold diagnostic

Metrics, percentile threshold, and oracle threshold diagnostic.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict:
    """คำนวณ metric ครบชุดจาก score/label/threshold ที่ให้มา — คืน dict เดียว
    รวมทั้ง classification metrics มาตรฐาน (AUC, AP, accuracy, precision,
    recall, F1) และ metric เชิงปฏิบัติงานจริง (auto-clear rate, escape
    rate, residual false-clear rate) ที่สื่อความหมายกับผู้ใช้งานจริงมากกว่า
    (เช่น ใน context QC/inspection line)

    Compute the full metric set from the given scores/labels/threshold —
    returns a single dict with both standard classification metrics (AUC,
    AP, accuracy, precision, recall, F1) and operationally meaningful
    metrics (auto-clear rate, escape rate, residual false-clear rate) that
    are easier to reason about in a real QC/inspection-line context.
    """
    pred = (scores >= threshold).astype(int)
    gt   = y_true
    fpr, tpr, _ = roc_curve(gt, scores)
    auc = roc_auc_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')
    ap  = average_precision_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')

    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_flagged = tn + fp + fn + tp
    # auto_clear_rate: สัดส่วนของภาพปกติทั้งหมดที่ระบบปล่อยผ่านถูกต้อง (ไม่ต้องตรวจซ้ำด้วยคน)
    # auto_clear_rate: fraction of all normal images the system correctly clears without human review
    auto_clear_rate = float(tn / n_flagged) if n_flagged > 0 else float('nan')
    # escape_rate: สัดส่วนของของเสียจริงที่หลุดรอดผ่านไปได้ (ตัวเลขที่อันตรายที่สุดในงาน QC)
    # escape_rate: fraction of true defects that slip through undetected (the most dangerous number in QC)
    escape_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else float('nan')
    # residual_fcr: ในบรรดาภาพที่ระบบตีว่าผิดปกติ สัดส่วนที่จริงๆ แล้วเป็นของดี (false alarm)
    # residual_fcr: of the images the system flags as anomalous, the fraction that were actually good (false alarms)
    residual_fcr = float(fp / (fp + tp)) if (fp + tp) > 0 else float('nan')

    # นับ confusion matrix แบบ 4 ช่อง เขียนเป็น (actual, predicted) โดย
    # T=anomaly (label 1), F=normal (label 0) — ตัวหน้าคือค่าจริง ตัวหลัง
    # คือค่าที่โมเดลทำนาย:
    #   TT = actual anomaly, predicted anomaly   -> True Positive  (tp)
    #   TF = actual anomaly, predicted normal    -> False Negative (fn)
    #   FT = actual normal,  predicted anomaly   -> False Positive (fp)
    #   FF = actual normal,  predicted normal    -> True Negative  (tn)
    # แปลงเป็น int ธรรมดา (ไม่ใช่ numpy int64) ให้ json.dump() serialize
    # ได้ตรงๆ โดยไม่ต้อง custom encoder
    #
    # 4-way confusion matrix counts, written as (actual, predicted) where
    # T=anomaly (label 1), F=normal (label 0) — first letter is ground
    # truth, second is the model's prediction:
    #   TT = actual anomaly, predicted anomaly   -> True Positive  (tp)
    #   TF = actual anomaly, predicted normal    -> False Negative (fn)
    #   FT = actual normal,  predicted anomaly   -> False Positive (fp)
    #   FF = actual normal,  predicted normal    -> True Negative  (tn)
    # Cast to plain int (not numpy int64) so json.dump() can serialize
    # them directly without a custom encoder.
    tt, tf, ft, ff = int(tp), int(fn), int(fp), int(tn)

    return dict(
        auc=auc, ap=ap,
        acc      = float(accuracy_score(gt, pred)),
        precision= float(precision_score(gt, pred, zero_division=0)),
        recall   = float(recall_score(gt, pred, zero_division=0)),
        f1       = float(f1_score(gt, pred, zero_division=0)),
        cm       = cm,
        tt=tt, tf=tf, ft=ft, ff=ff,
        auto_clear_rate = auto_clear_rate,
        escape_rate     = escape_rate,
        residual_fcr    = residual_fcr,
        fpr=fpr, tpr=tpr,
        gt=gt, pred=pred, scores=scores,
    )


def select_percentile_threshold(val_scores: np.ndarray, val_y: np.ndarray, cfg) -> float:
    """Threshold สำหรับ deployment: percentile ของ score ภาพ*ปกติ*ใน validation set

    Deployment threshold: percentile of the validation *normal* scores.
    """
    val_normal_scores = val_scores[val_y == 0]
    return float(np.percentile(val_normal_scores, cfg.THRESHOLD_PERCENTILE))


def oracle_threshold_diagnostic(val_scores: np.ndarray, val_y: np.ndarray) -> Tuple[float, float]:
    """สำหรับ diagnostic เท่านั้น: threshold ที่ให้ max-F1 บน Val (ใช้ label
    ความผิดปกติของ validation set โดยตรง — **ไม่ได้ใช้**ในตัวเลขที่รายงาน
    เป็นผลจริง เพราะการรู้ label ล่วงหน้าแบบนี้ไม่ใช่สถานการณ์ deployment จริง)

    Diagnostic only: max-F1 threshold on Val (uses val anomaly labels,
    NOT used for reported metrics).
    """
    precisions, recalls, thresholds_candidates = precision_recall_curve(val_y, val_scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    oracle_threshold = float(thresholds_candidates[best_idx])
    oracle_f1        = float(f1_scores[best_idx])
    return oracle_threshold, oracle_f1