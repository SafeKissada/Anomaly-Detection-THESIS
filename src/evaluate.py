"""Metric ต่างๆ, percentile threshold, และ oracle threshold diagnostic

Metrics, percentile threshold, and oracle threshold diagnostic.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)


def compute_metrics_from_predictions(gt: np.ndarray, pred: np.ndarray,
                                      scores: np.ndarray = None) -> Dict:
    """เวอร์ชัน core ของ compute_metrics(): รับ prediction (0/1) ตรงๆ
    ไม่ผ่าน threshold logic เลย — แยกออกมาจาก compute_metrics() เพื่อให้
    naive baseline (always_normal / always_anomaly / random_prior) เรียก
    ใช้ได้ตรงๆ โดยไม่ต้อง "หลอก" ฟังก์ชันเดิมด้วย score/threshold ปลอม
    (เช่น score=0, threshold=0.5) ซึ่งเสี่ยงเกิดบั๊กแบบ two-code-path-
    out-of-sync เหมือนที่เคยเจอใน engine.py (silent MSE fallback)

    ถ้า scores=None (ไม่มี ranking-based score ที่มีความหมาย ผูกกับ pred
    แบบต่อเนื่อง เช่น naive baseline ที่ pred เป็นค่าคงที่หรือสุ่มแบบไม่มี
    score รองรับ) ฟังก์ชันจะคืน auc=ap=NaN และ fpr=tpr=array ว่าง ตามหลัก
    ที่ว่า ranking-based metric ไม่มีความหมายกับ fixed/random decision
    rule (AUC/AP วัดคุณภาพการจัดอันดับของ score ต่อเนื่อง ไม่ใช่คุณภาพของ
    prediction 0/1 เดี่ยวๆ)

    Core version of compute_metrics(): takes predictions (0/1) directly,
    bypassing threshold logic entirely — split out from compute_metrics()
    so naive baselines (always_normal / always_anomaly / random_prior)
    can call it directly without "faking" the original function with a
    dummy score/threshold pair (e.g. score=0, threshold=0.5), which risks
    the same kind of two-code-path desync bug seen before in engine.py
    (the silent MSE fallback).

    If scores=None (no meaningful ranking-based score tied continuously
    to pred — e.g. naive baselines where pred is a constant or random
    value with no backing score), this returns auc=ap=NaN and
    fpr=tpr=empty arrays, per the principle that ranking-based metrics
    are meaningless for a fixed/random decision rule (AUC/AP measure the
    ranking quality of a continuous score, not the quality of a single
    0/1 prediction).
    """
    if scores is not None:
        fpr, tpr, _ = roc_curve(gt, scores)
        auc = roc_auc_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')
        ap  = average_precision_score(gt, scores) if len(np.unique(gt)) == 2 else float('nan')
    else:
        # ไม่มี score ต่อเนื่อง (naive baseline) — ranking metric ไม่มี
        # นิยาม บังคับ NaN ตรงๆ ไม่เดา ไม่คำนวณ roc_curve() ทิ้งเปล่าๆ
        #
        # No continuous score (naive baseline) — ranking metrics are
        # undefined. Force NaN directly; don't guess, and don't waste a
        # roc_curve() call for nothing.
        fpr, tpr = np.array([]), np.array([])
        auc = float('nan')
        ap  = float('nan')

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


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict:
    """คำนวณ metric ครบชุดจาก score/label/threshold ที่ให้มา — คืน dict เดียว
    รวมทั้ง classification metrics มาตรฐาน (AUC, AP, accuracy, precision,
    recall, F1) และ metric เชิงปฏิบัติงานจริง (auto-clear rate, escape
    rate, residual false-clear rate) ที่สื่อความหมายกับผู้ใช้งานจริงมากกว่า
    (เช่น ใน context QC/inspection line)

    เป็น wrapper บาง ๆ รอบ compute_metrics_from_predictions(): แปลง score
    ต่อเนื่องเป็น prediction 0/1 ด้วย threshold ก่อน แล้วส่งต่อให้ core
    function คำนวณ — ตัวโค้ดคำนวณจริงอยู่ที่เดียว ไม่มี logic ซ้ำซ้อน

    Compute the full metric set from the given scores/labels/threshold —
    returns a single dict with both standard classification metrics (AUC,
    AP, accuracy, precision, recall, F1) and operationally meaningful
    metrics (auto-clear rate, escape rate, residual false-clear rate) that
    are easier to reason about in a real QC/inspection-line context.

    A thin wrapper around compute_metrics_from_predictions(): converts the
    continuous score into a 0/1 prediction via the threshold, then hands
    off to the core function — the actual computation lives in one place,
    no duplicated logic.
    """
    pred = (scores >= threshold).astype(int)
    return compute_metrics_from_predictions(y_true, pred, scores=scores)


def compute_naive_baseline_metrics(y_true: np.ndarray, seed: int) -> Dict[str, Dict]:
    """คำนวณ metric ของ naive baseline 3 แบบ สำหรับเทียบกับผลโมเดลจริง
    (คำนวณเฉพาะบน val/test เท่านั้น — ห้ามเรียกกับ train split เพราะ
    train ของ repo นี้มีแต่ภาพ normal โดยการออกแบบ ทุก field จะกลายเป็น
    NaN/0 ที่ไม่มีความหมายไปหมด)

    - always_normal  : ไม่ flag เลย (pred=0 ทุกภาพ) — จำลอง "ไม่ตรวจอะไรเลย ปล่อยผ่านหมด"
    - always_anomaly : flag ทุกภาพ (pred=1 ทุกภาพ) — จำลอง "ตีว่าเสียหมดทุกชิ้น ให้คนตรวจซ้ำทุกใบ"
    - random_prior   : สุ่ม pred=1 ด้วยความน่าจะเป็น = สัดส่วน anomaly จริงใน y_true ที่ส่งเข้ามา
                        (stratified ตาม prior ของ split นั้นๆ)

    ทั้ง 3 แบบเรียก compute_metrics_from_predictions(scores=None) เสมอ
    → auc/ap เป็น NaN เสมอ ตามที่ตกลงไว้ (ranking metric ไม่มีความหมาย
    กับ decision rule แบบคงที่/สุ่ม)

    random_prior ใช้ np.random.RandomState(seed) แยกต่างหากของตัวเอง
    ไม่แตะ global RNG ที่ set_seed(cfg.SEED) ตั้งไว้ตอนต้น pipeline —
    ไม่กระทบ reproducibility ของการเทรน AE หรือของ split เดิมแต่อย่างใด

    Compute metrics for 3 naive baselines to compare against the real
    model's results (val/test only — never call this on the train split,
    since this repo's train set is normal-only by design; every field
    would collapse into a meaningless NaN/0).

    - always_normal  : never flags (pred=0 for every image) — simulates "inspect nothing, pass everything"
    - always_anomaly : flags every image (pred=1 for every image) — simulates "reject everything, re-inspect every unit by hand"
    - random_prior   : predicts 1 with probability = the true anomaly proportion in the given y_true
                        (stratified by that split's prior)

    All three always call compute_metrics_from_predictions(scores=None)
    → auc/ap are always NaN, per the earlier decision (ranking metrics
    are meaningless for a fixed/random decision rule).

    random_prior uses its own separate np.random.RandomState(seed) — it
    does not touch the global RNG that set_seed(cfg.SEED) configured at
    the start of the pipeline, so it has zero effect on AE training
    reproducibility or on the existing data split.
    """
    n = len(y_true)

    # แบบที่ 1: Always Normal — pred เป็น 0 ทุกตัว
    # Baseline 1: Always Normal — pred is 0 for every sample
    pred_always_normal = np.zeros(n, dtype=int)

    # แบบที่ 2: Always Anomaly — pred เป็น 1 ทุกตัว
    # Baseline 2: Always Anomaly — pred is 1 for every sample
    pred_always_anomaly = np.ones(n, dtype=int)

    # แบบที่ 3: Random / Prior-based — สุ่มตามสัดส่วน anomaly จริงของ split นี้
    # ต้อง log seed ที่ใช้ไว้เสมอ (ผ่าน caller เป็นคนเก็บ) เพื่อให้ผลสุ่ม
    # reproduce ซ้ำได้ข้าม run
    #
    # Baseline 3: Random / Prior-based — sample 1s at the true anomaly
    # rate for this split. The seed used must always be logged by the
    # caller so the random result reproduces exactly across runs.
    prior = float(np.mean(y_true))
    rng = np.random.RandomState(seed)
    pred_random = (rng.random_sample(n) < prior).astype(int)

    return {
        'always_normal':  compute_metrics_from_predictions(y_true, pred_always_normal,  scores=None),
        'always_anomaly': compute_metrics_from_predictions(y_true, pred_always_anomaly, scores=None),
        'random_prior':   compute_metrics_from_predictions(y_true, pred_random,         scores=None),
    }


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