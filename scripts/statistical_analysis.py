"""Bootstrap confidence intervals, paired tests, and calibration analysis."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import compute_all_metrics

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - fallback for older scipy
    binomtest = None


MODEL_PATHS = {
    "PREG-Net": "results/preg_net/test_predictions.csv",
    "FETA-Transformer": "results/feta_transformer/test_predictions.csv",
    "Ensemble": "results/ensemble/test_predictions.csv",
    "Learned Ensemble": "results/ensemble_learned/test_predictions.csv",
    "Torch Logistic Regression": "results/baselines/torch_logistic_regression/test_predictions.csv",
    "MLP": "results/baselines/torch_mlp/test_predictions.csv",
    "Sklearn Logistic Regression": "results/baselines/sklearn_logistic_regression/test_predictions.csv",
    "Random Forest": "results/baselines/random_forest/test_predictions.csv",
    "HistGradientBoosting": "results/baselines/hist_gradient_boosting/test_predictions.csv",
    "XGBoost": "results/baselines/xgboost/test_predictions.csv",
}

METRICS = [
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs and paired statistical comparisons."
    )
    parser.add_argument("--out-dir", default="results/statistical_analysis")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def load_predictions() -> Dict[str, pd.DataFrame]:
    predictions = {}
    for name, path_str in MODEL_PATHS.items():
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"patient_id", "label", "probability", "prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        predictions[name] = df.sort_values("patient_id").reset_index(drop=True)

    if not predictions:
        raise RuntimeError("No prediction files found.")

    reference = next(iter(predictions.values()))
    patient_ids = reference["patient_id"].tolist()
    labels = reference["label"].to_numpy(dtype=int)
    for name, df in predictions.items():
        if df["patient_id"].tolist() != patient_ids:
            raise ValueError(f"{name} patient IDs do not match reference order.")
        if not np.array_equal(df["label"].to_numpy(dtype=int), labels):
            raise ValueError(f"{name} labels do not match reference labels.")

    return predictions


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def fast_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(y_prob)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fast_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float(np.sum(precision[sorted_true == 1]) / n_pos)


def fast_metric_values(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )

    return {
        "auroc": fast_auroc(y_true, y_prob),
        "auprc": fast_auprc(y_true, y_prob),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "precision": precision,
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    scores: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    n = len(y_true)

    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        sample_prob = y_prob[idx]
        metrics = fast_metric_values(sample_true, sample_prob)
        for metric in METRICS:
            value = metrics[metric]
            if not np.isnan(value):
                scores[metric].append(float(value))

    cis = {}
    for metric, values in scores.items():
        if not values:
            cis[metric] = (float("nan"), float("nan"))
            continue
        cis[metric] = (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
    return cis


def metric_ci_rows(
    predictions: Dict[str, pd.DataFrame],
    iterations: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows = []
    for offset, (name, df) in enumerate(predictions.items()):
        y_true = df["label"].to_numpy(dtype=int)
        y_prob = df["probability"].to_numpy(dtype=float)
        point = compute_all_metrics(y_true, y_prob)
        cis = bootstrap_metric_ci(y_true, y_prob, iterations, seed + offset)

        for metric in METRICS:
            lower, upper = cis[metric]
            rows.append({
                "model": name,
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_bootstrap": iterations,
            })
    return rows


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)

    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def mcnemar_rows(predictions: Dict[str, pd.DataFrame]) -> List[Dict[str, object]]:
    rows = []
    for name_a, name_b in combinations(predictions, 2):
        df_a = predictions[name_a]
        df_b = predictions[name_b]
        labels = df_a["label"].to_numpy(dtype=int)
        pred_a = df_a["prediction"].to_numpy(dtype=int)
        pred_b = df_b["prediction"].to_numpy(dtype=int)

        correct_a = pred_a == labels
        correct_b = pred_b == labels
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))
        discordant = b + c
        statistic = (
            ((abs(b - c) - 1) ** 2 / discordant) if discordant > 0 else 0.0
        )
        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "a_correct_b_wrong": b,
            "a_wrong_b_correct": c,
            "discordant": discordant,
            "mcnemar_statistic_corrected": float(statistic),
            "p_value_exact": exact_mcnemar_pvalue(b, c),
        })
    rows.sort(key=lambda row: row["p_value_exact"])
    return rows


def auc_value(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return fast_auroc(y_true, y_prob)


def bootstrap_auroc_diff_rows(
    predictions: Dict[str, pd.DataFrame],
    iterations: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows = []
    rng = np.random.default_rng(seed)
    names = list(predictions)
    labels = predictions[names[0]]["label"].to_numpy(dtype=int)
    n = len(labels)

    for pair_idx, (name_a, name_b) in enumerate(combinations(names, 2)):
        prob_a = predictions[name_a]["probability"].to_numpy(dtype=float)
        prob_b = predictions[name_b]["probability"].to_numpy(dtype=float)
        observed = auc_value(labels, prob_a) - auc_value(labels, prob_b)

        diffs = []
        pair_rng = np.random.default_rng(seed + 1000 + pair_idx)
        for _ in range(iterations):
            idx = pair_rng.integers(0, n, size=n)
            sample_true = labels[idx]
            if len(np.unique(sample_true)) < 2:
                continue
            diffs.append(
                auc_value(sample_true, prob_a[idx])
                - auc_value(sample_true, prob_b[idx])
            )
        diffs_arr = np.asarray(diffs, dtype=float)
        if len(diffs_arr) == 0:
            ci_lower = float("nan")
            ci_upper = float("nan")
            p_two_sided = float("nan")
        else:
            ci_lower = float(np.percentile(diffs_arr, 2.5))
            ci_upper = float(np.percentile(diffs_arr, 97.5))
            p_two_sided = float(
                2 * min(np.mean(diffs_arr <= 0), np.mean(diffs_arr >= 0))
            )
            p_two_sided = min(1.0, p_two_sided)

        rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "auroc_a_minus_b": float(observed),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value_bootstrap": p_two_sided,
            "n_bootstrap": len(diffs),
            "method_note": "Bootstrap paired AUROC difference; documented alternative to DeLong.",
        })

    rows.sort(key=lambda row: row["p_value_bootstrap"])
    return rows


def calibration_rows(
    predictions: Dict[str, pd.DataFrame],
    n_bins: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metric_rows = []
    bin_rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for name, df in predictions.items():
        y_true = df["label"].to_numpy(dtype=int)
        y_prob = df["probability"].to_numpy(dtype=float)
        brier = float(np.mean((y_prob - y_true) ** 2))

        ece = 0.0
        mce = 0.0
        for bin_idx in range(n_bins):
            left = edges[bin_idx]
            right = edges[bin_idx + 1]
            if bin_idx == n_bins - 1:
                mask = (y_prob >= left) & (y_prob <= right)
            else:
                mask = (y_prob >= left) & (y_prob < right)
            n_bin = int(np.sum(mask))

            if n_bin == 0:
                mean_probability = float("nan")
                observed_rate = float("nan")
                abs_error = float("nan")
            else:
                mean_probability = float(np.mean(y_prob[mask]))
                observed_rate = float(np.mean(y_true[mask]))
                abs_error = abs(observed_rate - mean_probability)
                ece += (n_bin / len(y_true)) * abs_error
                mce = max(mce, abs_error)

            bin_rows.append({
                "model": name,
                "bin": bin_idx,
                "bin_lower": float(left),
                "bin_upper": float(right),
                "n": n_bin,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "abs_calibration_error": abs_error,
            })

        metric_rows.append({
            "model": name,
            "brier_score": brier,
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "mean_probability": float(np.mean(y_prob)),
            "observed_prevalence": float(np.mean(y_true)),
            "n_bins": n_bins,
        })

    metric_rows.sort(key=lambda row: row["brier_score"])
    return metric_rows, bin_rows


def write_calibration_plot(
    path: Path,
    calibration_bin_rows: List[Dict[str, object]],
    model_names: List[str],
) -> None:
    path.with_suffix(".plot_error.txt").unlink(missing_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional plotting fallback
        write_calibration_svg(
            path.with_suffix(".svg"),
            calibration_bin_rows,
            model_names,
        )
        return

    n_models = len(model_names)
    n_cols = 2
    n_rows = int(math.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), squeeze=False)
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in calibration_bin_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    for idx, name in enumerate(model_names):
        ax = axes[idx // n_cols][idx % n_cols]
        rows = [row for row in by_model[name] if int(row["n"]) > 0]
        x = [float(row["mean_probability"]) for row in rows]
        y = [float(row["observed_rate"]) for row in rows]
        sizes = [max(20, int(row["n"]) * 8) for row in rows]
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
        ax.plot(x, y, color="#1f77b4", linewidth=1.5)
        ax.scatter(x, y, s=sizes, color="#1f77b4", alpha=0.8)
        ax.set_title(name)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.grid(alpha=0.25)

    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_calibration_svg(
    path: Path,
    calibration_bin_rows: List[Dict[str, object]],
    model_names: List[str],
    note: str | None = None,
) -> None:
    n_models = len(model_names)
    n_cols = 2
    n_rows = int(math.ceil(n_models / n_cols))
    cell_w = 460
    cell_h = 340
    width = n_cols * cell_w
    height = n_rows * cell_h + (26 if note else 0)
    plot_left = 58
    plot_top = 48
    plot_w = 340
    plot_h = 230

    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in calibration_bin_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    def x_pos(origin_x: int, value: float) -> float:
        return origin_x + plot_left + value * plot_w

    def y_pos(origin_y: int, value: float) -> float:
        return origin_y + plot_top + (1.0 - value) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#222}'
        '.axis{stroke:#222;stroke-width:1}'
        '.grid{stroke:#ddd;stroke-width:1}'
        '.diag{stroke:#777;stroke-width:1;stroke-dasharray:5 5}'
        '.line{fill:none;stroke:#1f77b4;stroke-width:2}'
        '.point{fill:#1f77b4;fill-opacity:.82}'
        '</style>',
    ]

    for idx, name in enumerate(model_names):
        origin_x = (idx % n_cols) * cell_w
        origin_y = (idx // n_cols) * cell_h
        rows = [row for row in by_model.get(name, []) if int(row["n"]) > 0]
        rows.sort(key=lambda row: float(row["mean_probability"]))

        lines.append(f'<text x="{origin_x + 18}" y="{origin_y + 24}" font-size="15" font-weight="600">{html.escape(name)}</text>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x_pos(origin_x, tick)
            y = y_pos(origin_y, tick)
            lines.append(f'<line class="grid" x1="{x:.1f}" y1="{origin_y + plot_top}" x2="{x:.1f}" y2="{origin_y + plot_top + plot_h}"/>')
            lines.append(f'<line class="grid" x1="{origin_x + plot_left}" y1="{y:.1f}" x2="{origin_x + plot_left + plot_w}" y2="{y:.1f}"/>')
            lines.append(f'<text x="{x - 8:.1f}" y="{origin_y + plot_top + plot_h + 20}" font-size="10">{tick:.2g}</text>')
            lines.append(f'<text x="{origin_x + plot_left - 34}" y="{y + 3:.1f}" font-size="10">{tick:.2g}</text>')

        x0 = origin_x + plot_left
        y0 = origin_y + plot_top
        x1 = x0 + plot_w
        y1 = y0 + plot_h
        lines.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
        lines.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}"/>')
        lines.append(f'<line class="diag" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}"/>')

        points = [
            (x_pos(origin_x, float(row["mean_probability"])), y_pos(origin_y, float(row["observed_rate"])))
            for row in rows
        ]
        if points:
            point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline class="line" points="{point_str}"/>')
        for row, (x, y) in zip(rows, points):
            radius = 3.0 + min(7.0, math.sqrt(int(row["n"])))
            lines.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}"/>')

        lines.append(
            f'<text x="{origin_x + plot_left + 72}" y="{origin_y + plot_top + plot_h + 39}" '
            'font-size="11">Mean predicted probability</text>'
        )
        lines.append(
            f'<text x="{origin_x + 12}" y="{origin_y + plot_top + 145}" '
            'font-size="11" transform="rotate(-90 '
            f'{origin_x + 12} {origin_y + plot_top + 145})">Observed event rate</text>'
        )

    if note:
        lines.append(
            f'<text x="16" y="{height - 8}" font-size="11" fill="#555">{html.escape(note)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_ci_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    lines = [
        "# Bootstrap 95% Confidence Intervals",
        "",
        "| Model | Metric | Estimate | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for model, model_rows in by_model.items():
        for row in model_rows:
            lines.append(
                f"| {model} | {row['metric']} | {row['estimate']:.4f} | "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    path: Path,
    metric_rows: List[Dict[str, object]],
    mcnemar_rows_: List[Dict[str, object]],
    auroc_diff_rows: List[Dict[str, object]],
    calibration_metric_rows: List[Dict[str, object]],
    n_models: int,
    iterations: int,
    n_patients: int,
) -> None:
    lines = [
        "# Statistical Analysis Summary",
        "",
        f"- Models analyzed: {n_models}",
        f"- Test patients: {n_patients}",
        f"- Bootstrap iterations requested: {iterations}",
        "- AUROC comparison method: paired bootstrap difference, used as the documented DeLong alternative.",
        "",
        "## Bootstrap Confidence Intervals",
        "",
        "| Model | Metric | Estimate | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['estimate']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |"
        )

    lines.extend([
        "",
        "## Smallest McNemar P Values",
        "",
        "| Model A | Model B | Discordant | Exact p |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in mcnemar_rows_[:10]:
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | "
            f"{row['discordant']} | {row['p_value_exact']:.4f} |"
        )

    lines.extend([
        "",
        "## Smallest AUROC-Difference P Values",
        "",
        "| Model A | Model B | AUROC diff | 95% CI | Bootstrap p |",
        "| --- | --- | ---: | --- | ---: |",
    ])
    for row in auroc_diff_rows[:10]:
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | "
            f"{row['auroc_a_minus_b']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{row['p_value_bootstrap']:.4f} |"
        )

    lines.extend([
        "",
        "## Calibration Metrics",
        "",
        "| Model | Brier score | ECE | MCE | Mean prediction | Prevalence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in calibration_metric_rows:
        lines.append(
            f"| {row['model']} | {row['brier_score']:.4f} | "
            f"{row['expected_calibration_error']:.4f} | "
            f"{row['max_calibration_error']:.4f} | "
            f"{row['mean_probability']:.4f} | "
            f"{row['observed_prevalence']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    first_df = next(iter(predictions.values()))
    metric_rows = metric_ci_rows(predictions, args.iterations, args.seed)
    mcnemar = mcnemar_rows(predictions)
    auroc_diffs = bootstrap_auroc_diff_rows(predictions, args.iterations, args.seed)
    calibration_metrics, calibration_bins = calibration_rows(
        predictions,
        args.calibration_bins,
    )

    write_csv(out_dir / "metric_confidence_intervals.csv", metric_rows)
    write_csv(out_dir / "mcnemar_tests.csv", mcnemar)
    write_csv(out_dir / "auroc_bootstrap_comparisons.csv", auroc_diffs)
    write_csv(out_dir / "calibration_metrics.csv", calibration_metrics)
    write_csv(out_dir / "calibration_bins.csv", calibration_bins)
    write_metric_ci_markdown(out_dir / "metric_confidence_intervals.md", metric_rows)
    write_summary_markdown(
        out_dir / "summary.md",
        metric_rows,
        mcnemar,
        auroc_diffs,
        calibration_metrics,
        len(predictions),
        args.iterations,
        len(first_df),
    )
    write_calibration_plot(
        out_dir / "calibration_curves.png",
        calibration_bins,
        list(predictions),
    )

    (out_dir / "metric_confidence_intervals.json").write_text(
        json.dumps(metric_rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "mcnemar_tests.json").write_text(
        json.dumps(mcnemar, indent=2),
        encoding="utf-8",
    )
    (out_dir / "auroc_bootstrap_comparisons.json").write_text(
        json.dumps(auroc_diffs, indent=2),
        encoding="utf-8",
    )
    (out_dir / "calibration_metrics.json").write_text(
        json.dumps(calibration_metrics, indent=2),
        encoding="utf-8",
    )
    (out_dir / "calibration_bins.json").write_text(
        json.dumps(calibration_bins, indent=2),
        encoding="utf-8",
    )

    print(f"Analyzed {len(predictions)} models with {args.iterations} bootstrap iterations.")
    print(f"Saved statistical analysis outputs to: {out_dir}")
    print("\nTop AUROC comparisons by p-value:")
    for row in auroc_diffs[:5]:
        print(
            f"  {row['model_a']} vs {row['model_b']}: "
            f"diff={row['auroc_a_minus_b']:.4f}, "
            f"95% CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
            f"p={row['p_value_bootstrap']:.4f}"
        )


if __name__ == "__main__":
    main()
