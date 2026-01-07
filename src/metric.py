from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, cohen_kappa_score
)
from sklearn.metrics import roc_curve, precision_recall_curve


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    """
    Return a dict with common PTM-site metrics.
    y_true: [N] {0,1}
    y_prob: [N] in [0,1]
    """
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sn = tp / max(1, (tp + fn))
    sp = tn / max(1, (tn + fp))
    bacc = 0.5 * (sn + sp)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denom == 0 else (tp * tn - fp * fn) / denom

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    f1 = 0.0 if (precision + sn) == 0 else 2 * precision * sn / (precision + sn)

    # AUC/AP (handle degenerate case)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "BACC": bacc,
        "ACC": acc,
        "AUC": auc,
        "SN": sn,
        "SP": sp,
        "MCC": mcc,
        "F1": f1,
        "Precision": precision,
        "Kappa": kappa,
        "AP": ap,
        "thr": thr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def roc_pr_data(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = y_true.astype(int).reshape(-1)
    y_prob = y_prob.astype(float).reshape(-1)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return (fpr, tpr), (prec, rec)
