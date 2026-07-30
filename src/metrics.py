from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_safety_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_precision: float = 0.40,
) -> float:
    """Prefer the lowest false-negative rate subject to a precision floor."""
    candidates = np.linspace(0.05, 0.90, 172)
    rows: list[tuple[float, float, float, float]] = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0.0
        )
        rows.append((float(threshold), precision, recall, f2))

    eligible = [row for row in rows if row[1] >= minimum_precision]

    if eligible:
        # Highest recall, then F2, then higher precision.
        best = max(eligible, key=lambda row: (row[2], row[3], row[1]))
    else:
        best = max(rows, key=lambda row: row[3])

    return float(best[0])


def binary_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def confusion_matrix_frame(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["actual_normal", "actual_anomaly"],
        columns=["predicted_normal", "predicted_anomaly"],
    )


def measure_inference_ms_per_sample(
    probability_function: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    repeats: int = 250,
) -> float:
    """Measure single-observation latency rather than batch throughput."""
    sample = features[:1]
    probability_function(sample)

    start = perf_counter()
    for _ in range(repeats):
        probability_function(sample)
    elapsed = perf_counter() - start

    return float(elapsed * 1_000 / repeats)
