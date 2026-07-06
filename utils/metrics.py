"""
Evaluation metrics for the AVM pregnancy loss prediction models.

Provides:
  - :func:`compute_all_metrics` — AUROC, AUPRC, accuracy, sensitivity,
    specificity, F1, precision, recall from predictions and ground truth.
  - :func:`bootstrap_ci` — bootstrap confidence interval for any metric.
  - :func:`print_metrics_table` — formatted table for logging / display.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError:
    def accuracy_score(y_true, y_pred):
        return np.mean(np.asarray(y_true) == np.asarray(y_pred))

    def precision_score(y_true, y_pred, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)

    def recall_score(y_true, y_pred, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)

    def f1_score(y_true, y_pred, zero_division=0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        if precision + recall == 0:
            return zero_division
        return 2 * precision * recall / (precision + recall)

    def confusion_matrix(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

    def _average_ranks(values):
        values = np.asarray(values)
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return ranks

    def roc_auc_score(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("AUROC is undefined with one class.")
        ranks = _average_ranks(y_prob)
        pos_rank_sum = np.sum(ranks[y_true == 1])
        return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def average_precision_score(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        n_pos = np.sum(y_true == 1)
        if n_pos == 0:
            return 0.0
        order = np.argsort(-y_prob)
        sorted_true = y_true[order]
        tp = np.cumsum(sorted_true == 1)
        precision = tp / (np.arange(len(sorted_true)) + 1)
        return np.sum(precision[sorted_true == 1]) / n_pos


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: 1-D array of ground-truth labels (0 or 1).
        y_prob: 1-D array of predicted probabilities ∈ [0, 1].
        threshold: Decision threshold for converting probabilities to
            binary predictions (default 0.5).

    Returns:
        Dictionary containing:
            - ``auroc``: Area Under the Receiver Operating Characteristic curve.
            - ``auprc``: Area Under the Precision-Recall Curve.
            - ``accuracy``: Overall classification accuracy.
            - ``sensitivity`` (= recall): True Positive Rate.
            - ``specificity``: True Negative Rate.
            - ``f1``: F1 score.
            - ``precision``: Positive Predictive Value.
            - ``recall``: True Positive Rate (alias of sensitivity).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    # Handle edge cases (single class in y_true)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Specificity from confusion matrix
    if len(unique_classes) < 2:
        specificity = float("nan")
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "sensitivity": rec,
        "specificity": specificity,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a given metric.

    Resamples (with replacement) ``n_iterations`` times and returns the
    percentile-based confidence interval.

    Args:
        y_true: 1-D array of ground-truth labels.
        y_prob: 1-D array of predicted probabilities.
        metric_fn: Callable ``(y_true, y_prob) -> float`` that computes
            the metric of interest.
        n_iterations: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95 for 95 % CI).
        seed: Random seed for reproducibility.

    Returns:
        ``(lower, upper)`` bounds of the confidence interval.

    Example::

        lower, upper = bootstrap_ci(
            y_true, y_prob,
            metric_fn=roc_auc_score,
            n_iterations=2000,
        )
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        # Skip degenerate samples (single class)
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, yp))
        except Exception:
            continue

    if len(scores) == 0:
        return (float("nan"), float("nan"))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1.0 - alpha)))
    return lower, upper


def print_metrics_table(metrics_dict: Dict[str, float]) -> str:
    """Format a metrics dictionary as a human-readable table.

    Args:
        metrics_dict: Dictionary of metric names to values (as returned
            by :func:`compute_all_metrics`).

    Returns:
        A formatted multi-line string ready for printing or logging.

    Example::

        metrics = compute_all_metrics(y_true, y_prob)
        print(print_metrics_table(metrics))
    """
    # Determine column widths
    name_width = max(len(k) for k in metrics_dict) + 2
    lines = []
    lines.append("┌" + "─" * (name_width + 12) + "┐")
    lines.append(f"│ {'Metric':<{name_width}} {'Value':>8}   │")
    lines.append("├" + "─" * (name_width + 12) + "┤")

    for name, value in metrics_dict.items():
        if np.isnan(value):
            val_str = "     N/A"
        else:
            val_str = f"{value:>8.4f}"
        lines.append(f"│ {name:<{name_width}} {val_str}   │")

    lines.append("└" + "─" * (name_width + 12) + "┘")
    return "\n".join(lines)
