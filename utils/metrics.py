import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_prob: np.ndarray
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    y_prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob_pos)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
        "example_count": float(len(y_true)),
        "accuracy_score": float(accuracy_score(y_true, y_pred)),
        "recall_score": float(recall_score(y_true, y_pred)),
        "precision_score": float(precision_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_pos)),
        "precision_recall_auc": float(pr_auc),
    }
