from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_roc_auc(y_true, scores) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": safe_roc_auc(y_true, scores),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(predictions.sum()),
    }


def find_best_threshold(
    y_true,
    scores,
    beta: float = 2.0,
    min_precision: float | None = None,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        fbeta = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f2": float(fbeta),
                "alerts": int(predictions.sum()),
            }
        )

    curve = pd.DataFrame(rows)
    if min_precision is not None:
        candidates = curve[curve["precision"] >= min_precision]
        if candidates.empty:
            candidates = curve
    else:
        candidates = curve

    best = candidates.sort_values(["f2", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), curve


def precision_recall_frame(y_true, scores) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    padded_thresholds = np.r_[thresholds, np.nan]
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": padded_thresholds,
        }
    )


def confusion_frame(y_true, scores, threshold: float) -> pd.DataFrame:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["real_legitima", "real_fraude"],
        columns=["prevista_legitima", "prevista_fraude"],
    )


def cost_metrics(
    y_true,
    scores,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
    true_positive_review_cost: float = 0.0,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    total_cost = (
        fp * false_positive_cost
        + fn * false_negative_cost
        + tp * true_positive_review_cost
    )
    return {
        "threshold": float(threshold),
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost": float(false_negative_cost),
        "true_positive_review_cost": float(true_positive_review_cost),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(predictions.sum()),
        "total_cost": float(total_cost),
    }


def cost_curve(
    y_true,
    scores,
    false_positive_cost: float,
    false_negative_cost: float,
    true_positive_review_cost: float = 0.0,
) -> pd.DataFrame:
    rows = [
        cost_metrics(
            y_true,
            scores,
            threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            true_positive_review_cost=true_positive_review_cost,
        )
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    curve = pd.DataFrame(rows)
    curve["cost_per_transaction"] = curve["total_cost"] / max(len(y_true), 1)
    return curve


def best_cost_threshold(
    y_true,
    scores,
    false_positive_cost: float,
    false_negative_cost: float,
    true_positive_review_cost: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    curve = cost_curve(
        y_true,
        scores,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        true_positive_review_cost=true_positive_review_cost,
    )
    best = curve.sort_values(["total_cost", "false_negatives", "false_positives"]).iloc[0]
    return float(best["threshold"]), curve
