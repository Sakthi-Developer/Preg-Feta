"""Generate paper-ready figures and tables from saved experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_PREDICTIONS = {
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
RESEARCH_MODELS = ["PREG-Net", "FETA-Transformer", "Ensemble", "Learned Ensemble"]
MODEL_COLORS = {
    "PREG-Net": "#2f5fb3",
    "FETA-Transformer": "#c45a4a",
    "Ensemble": "#2f8f5b",
    "Learned Ensemble": "#7854a3",
    "Torch Logistic Regression": "#8a8a8a",
    "MLP": "#111111",
    "Sklearn Logistic Regression": "#d08c2d",
    "Random Forest": "#4b9aa8",
    "HistGradientBoosting": "#a75b8f",
    "XGBoost": "#6b7d2d",
}
METRIC_COLUMNS = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1", "precision"]
COMPARISON_ALIASES = {
    "Torch Logistic Regression": "Logistic Regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures and tables.")
    parser.add_argument("--patients-csv", default="data/generated/patients.csv")
    parser.add_argument("--scans-csv", default="data/generated/scans.csv")
    parser.add_argument("--out-dir", default="results/paper_artifacts")
    parser.add_argument("--comparison-csv", default="results/model_comparison.csv")
    parser.add_argument("--cv-summary-csv", default="results/cross_validation/summary.csv")
    parser.add_argument("--ci-csv", default="results/statistical_analysis/metric_confidence_intervals.csv")
    parser.add_argument("--calibration-svg", default="results/statistical_analysis/calibration_curves.svg")
    parser.add_argument("--explainability-dir", default="results/explainability")
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def write_markdown_table(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: List[Dict[str, object]], caption: str, label: str) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    align = "l" * len(columns)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(str(row.get(col, ""))) for col in columns) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def table_bundle(
    tables_dir: Path,
    stem: str,
    rows: List[Dict[str, object]],
    caption: str,
    label: str,
) -> None:
    write_csv(tables_dir / f"{stem}.csv", rows)
    write_markdown_table(tables_dir / f"{stem}.md", rows)
    write_latex_table(tables_dir / f"{stem}.tex", rows, caption, label)


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#202124}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#e1e5ea;stroke-width:1}'
        '.label{font-size:12px}.title{font-size:18px;font-weight:700}'
        '.subtitle{font-size:12px;fill:#555}.box{fill:#f7f9fc;stroke:#7b8794;stroke-width:1.2}'
        '.arrow{stroke:#333;stroke-width:1.5;marker-end:url(#arrow)}'
        '</style>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>',
    ]


def svg_text(x: float, y: float, text: object, size: int = 12, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{html.escape(str(text))}</text>'
    )


def write_architecture_svg(path: Path, model: str) -> None:
    width, height = 980, 360
    lines = svg_header(width, height)
    lines.append(svg_text(24, 32, f"{model} Architecture", 20, "700"))
    if model == "FETA-Transformer":
        boxes = [
            ("Longitudinal ultrasound\nFHR, CRL, GS, YSD", 40, 95),
            ("Modality-specific\nlinear projections", 235, 95),
            ("Continuous gestational\nage positional encoding", 430, 95),
            ("Transformer encoder\nself-attention over scans", 625, 95),
            ("Temporal attention pooling\n+ maternal conditioning", 820, 95),
        ]
        footer = "Output: pregnancy-loss probability; exported: temporal pooling weights and post-hoc maternal-to-time maps"
    else:
        boxes = [
            ("Patient graph nodes\nUS variables + maternal risk", 40, 95),
            ("Knowledge-guided edges\nphysiology + temporal links", 235, 95),
            ("Graph attention layers\npatient-specific messages", 430, 95),
            ("Attention graph readout\nnode importance", 625, 95),
            ("Classifier\nrisk probability", 820, 95),
        ]
        footer = "Output: pregnancy-loss probability; exported: node importance and edge attention rankings"

    box_w, box_h = 140, 95
    for label, x, y in boxes:
        lines.append(f'<rect class="box" x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6"/>')
        for idx, line in enumerate(label.split("\n")):
            lines.append(svg_text(x + box_w / 2, y + 34 + idx * 18, line, 12, "600" if idx == 0 else "400", "middle"))
    for (_, x, y), (_, x_next, _) in zip(boxes, boxes[1:]):
        lines.append(f'<line class="arrow" x1="{x + box_w + 10}" y1="{y + box_h / 2}" x2="{x_next - 10}" y2="{y + box_h / 2}"/>')
    lines.append(svg_text(40, 250, footer, 13, "500"))
    lines.append(svg_text(40, 292, "Synthetic first-trimester cohort: multimodal temporal scans plus maternal features", 12))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_predictions() -> Dict[str, pd.DataFrame]:
    predictions = {}
    for name, path_str in MODEL_PREDICTIONS.items():
        path = Path(path_str)
        if path.exists():
            predictions[name] = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
    return predictions


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return fpr, tpr


def pr_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(int(np.sum(y_true == 1)), 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision


def curve_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray, float]],
    diagonal: bool = False,
) -> None:
    width, height = 880, 620
    left, top, plot_w, plot_h = 80, 70, 560, 440
    lines = svg_header(width, height)
    lines.append(svg_text(24, 34, title, 20, "700"))
    for tick in np.linspace(0, 1, 6):
        x = left + tick * plot_w
        y = top + (1 - tick) * plot_h
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(svg_text(x, top + plot_h + 22, f"{tick:.1f}", 11, anchor="middle"))
        lines.append(svg_text(left - 16, y + 4, f"{tick:.1f}", 11, anchor="end"))
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    if diagonal:
        lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#999" stroke-width="1.2" stroke-dasharray="5 5"/>')

    for name, x_vals, y_vals, _score in curves:
        points = " ".join(
            f"{left + x * plot_w:.1f},{top + (1 - y) * plot_h:.1f}"
            for x, y in zip(x_vals, y_vals)
        )
        color = MODEL_COLORS.get(name, "#333")
        width_line = "2.6" if name in RESEARCH_MODELS else "1.5"
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width_line}" opacity="0.92"/>')

    lines.append(svg_text(left + plot_w / 2, top + plot_h + 52, x_label, 13, "600", "middle"))
    lines.append(
        f'<text x="24" y="{top + plot_h / 2:.1f}" font-size="13" font-weight="600" '
        f'text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>'
    )

    legend_x, legend_y = 670, 78
    for idx, (name, _x, _y, score) in enumerate(curves):
        y = legend_y + idx * 24
        color = MODEL_COLORS.get(name, "#333")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(svg_text(legend_x + 30, y + 4, f"{name} ({score:.3f})", 11))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_roc_pr_figures(figures_dir: Path, comparison: pd.DataFrame, predictions: Dict[str, pd.DataFrame]) -> None:
    auroc_by_model = comparison.set_index("model")["auroc"].to_dict()
    auprc_by_model = comparison.set_index("model")["auprc"].to_dict()
    roc_curves = []
    pr_curves = []
    for name, df in predictions.items():
        y_true = df["label"].to_numpy(dtype=int)
        y_score = df["probability"].to_numpy(dtype=float)
        fpr, tpr = roc_points(y_true, y_score)
        recall, precision = pr_points(y_true, y_score)
        comparison_name = COMPARISON_ALIASES.get(name, name)
        roc_curves.append((name, fpr, tpr, float(auroc_by_model.get(comparison_name, np.nan))))
        pr_curves.append((name, recall, precision, float(auprc_by_model.get(comparison_name, np.nan))))
    roc_curves.sort(key=lambda row: row[3], reverse=True)
    pr_curves.sort(key=lambda row: row[3], reverse=True)
    curve_svg(figures_dir / "fig_roc_curves.svg", "ROC Curves On Fixed Test Split", "False positive rate", "True positive rate", roc_curves, diagonal=True)
    curve_svg(figures_dir / "fig_precision_recall_curves.svg", "Precision-Recall Curves On Fixed Test Split", "Recall", "Precision", pr_curves, diagonal=False)


def write_confusion_matrix_svg(figures_dir: Path, predictions: Dict[str, pd.DataFrame]) -> None:
    names = list(predictions)
    cell = 62
    panel_w, panel_h = 205, 170
    width, height = 1080, 410
    lines = svg_header(width, height)
    lines.append(svg_text(24, 32, "Confusion Matrices On Fixed Test Split", 20, "700"))
    max_count = 1
    counts_by_model = {}
    for name, df in predictions.items():
        y = df["label"].to_numpy(dtype=int)
        p = df["prediction"].to_numpy(dtype=int)
        tn = int(np.sum((y == 0) & (p == 0)))
        fp = int(np.sum((y == 0) & (p == 1)))
        fn = int(np.sum((y == 1) & (p == 0)))
        tp = int(np.sum((y == 1) & (p == 1)))
        counts = np.array([[tn, fp], [fn, tp]])
        max_count = max(max_count, int(counts.max()))
        counts_by_model[name] = counts

    for idx, name in enumerate(names):
        col, row = idx % 5, idx // 5
        x0 = 35 + col * panel_w
        y0 = 70 + row * panel_h
        lines.append(svg_text(x0, y0 - 10, name, 12, "700"))
        counts = counts_by_model[name]
        for r in range(2):
            for c in range(2):
                value = int(counts[r, c])
                intensity = value / max_count
                fill = interpolate_color("#f6fbff", "#2f5fb3", intensity)
                x = x0 + 40 + c * cell
                y = y0 + 20 + r * cell
                lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
                color = "#ffffff" if intensity > 0.55 else "#202124"
                lines.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" font-size="18" font-weight="700" text-anchor="middle" fill="{color}">{value}</text>')
        lines.append(svg_text(x0 + 40 + cell, y0 + 20 + 2 * cell + 18, "Predicted", 10, anchor="middle"))
        lines.append(svg_text(x0 + 10, y0 + 20 + cell, "Actual", 10, anchor="middle"))
    lines.append("</svg>")
    (figures_dir / "fig_confusion_matrices.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpolate_color(low: str, high: str, frac: float) -> str:
    frac = max(0.0, min(1.0, frac))
    l = tuple(int(low[i:i + 2], 16) for i in (1, 3, 5))
    h = tuple(int(high[i:i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(int(lv + (hv - lv) * frac) for lv, hv in zip(l, h))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def write_heatmap_svg(path: Path, title: str, df: pd.DataFrame, value_col: str) -> None:
    weeks = sorted(df["gestational_week"].dropna().astype(int).unique().tolist())
    outcomes = [("ongoing", 0), ("loss", 1)]
    cell_w, cell_h = 54, 56
    left, top = 130, 72
    width = left + len(weeks) * cell_w + 80
    height = top + len(outcomes) * cell_h + 95
    lines = svg_header(width, height)
    lines.append(svg_text(24, 34, title, 20, "700"))
    max_val = max(float(df[value_col].max()), 1e-8)
    for j, week in enumerate(weeks):
        lines.append(svg_text(left + j * cell_w + cell_w / 2, top - 15, week, 11, "600", "middle"))
    for i, (outcome, label) in enumerate(outcomes):
        lines.append(svg_text(left - 16, top + i * cell_h + cell_h / 2 + 4, outcome, 12, "600", "end"))
        for j, week in enumerate(weeks):
            subset = df[(df["label"] == label) & (df["gestational_week"] == week)]
            value = float(subset[value_col].iloc[0]) if not subset.empty else float("nan")
            fill = "#f2f3f5" if math.isnan(value) else interpolate_color("#f6fbff", "#2f5fb3", value / max_val)
            x = left + j * cell_w
            y = top + i * cell_h
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            label_text = "" if math.isnan(value) else f"{value:.2f}"
            lines.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, label_text, 11, "600", "middle"))
    lines.append(svg_text(left + len(weeks) * cell_w / 2, height - 32, "Rounded gestational week", 12, "600", "middle"))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bar_svg(
    path: Path,
    title: str,
    rows: List[Tuple[str, float]],
    x_label: str,
    color: str = "#2f5fb3",
) -> None:
    width = 850
    row_h = 34
    top = 70
    left = 270
    plot_w = 500
    height = top + len(rows) * row_h + 70
    max_val = max([value for _, value in rows] + [1e-8])
    lines = svg_header(width, height)
    lines.append(svg_text(24, 34, title, 20, "700"))
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_h
        bar_w = value / max_val * plot_w
        lines.append(svg_text(left - 12, y + 18, label, 11, "500", "end"))
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" fill="{color}" opacity="0.86"/>')
        lines.append(svg_text(left + bar_w + 8, y + 16, f"{value:.3f}", 11))
    lines.append(svg_text(left + plot_w / 2, height - 24, x_label, 12, "600", "middle"))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_status_svg(path: Path) -> None:
    width, height = 820, 280
    lines = svg_header(width, height)
    lines.append(svg_text(24, 34, "Ablation Results Status", 20, "700"))
    lines.append(svg_text(36, 82, "Ablation switches and runs are not available in the saved experiment set.", 14, "600"))
    lines.append(svg_text(36, 112, "Generated artifact is a status marker only; do not report it as an ablation result.", 12))
    items = [
        "FETA: no positional encoding, no attention pooling, no maternal cross-attention",
        "PREG-Net: no knowledge edges, random graph, no temporal edges, no edge attention",
        "Ensemble: fusion variants beyond average/learned fusion",
    ]
    for idx, item in enumerate(items):
        lines.append(svg_text(58, 154 + idx * 32, item, 12))
        lines.append(f'<circle cx="40" cy="{150 + idx * 32}" r="5" fill="#c45a4a"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_demographics_table(patients: pd.DataFrame, scans: pd.DataFrame) -> List[Dict[str, object]]:
    merged_scan_counts = scans.groupby("patient_id").size().rename("scan_count")
    patients = patients.merge(merged_scan_counts, on="patient_id", how="left")
    patients["ivf"] = (patients["conception"].astype(str).str.upper() == "IVF").astype(float)
    patients["singleton_float"] = patients["singleton"].astype(float)
    first_ga = scans.groupby("patient_id")["gestational_age_weeks"].min().rename("first_ga")
    last_ga = scans.groupby("patient_id")["gestational_age_weeks"].max().rename("last_ga")
    patients = patients.merge(first_ga, on="patient_id").merge(last_ga, on="patient_id")

    specs = [
        ("Patients, n", "count", None),
        ("Loss rate, %", "binary_pct", "label"),
        ("Scans per patient", "mean_sd", "scan_count"),
        ("First GA, weeks", "mean_sd", "first_ga"),
        ("Latest GA, weeks", "mean_sd", "last_ga"),
        ("Maternal age, years", "mean_sd", "age"),
        ("BMI", "mean_sd", "bmi"),
        ("Parity", "mean_sd", "parity"),
        ("Gravidity", "mean_sd", "gravidity"),
        ("Previous loss, %", "binary_pct", "previous_loss"),
        ("IVF conception, %", "binary_pct", "ivf"),
        ("Singleton, %", "binary_pct", "singleton_float"),
    ]
    rows = []
    groups = [("Overall", patients), ("Ongoing", patients[patients["label"] == 0]), ("Loss", patients[patients["label"] == 1])]
    for label, kind, col in specs:
        row = {"Characteristic": label}
        for group_name, df in groups:
            if kind == "count":
                row[group_name] = str(len(df))
            elif kind == "binary_pct":
                row[group_name] = f"{df[col].mean() * 100:.1f}"
            else:
                row[group_name] = f"{df[col].mean():.2f} +/- {df[col].std():.2f}"
        rows.append(row)
    return rows


def rounded_model_comparison(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    rows = []
    for row in df.itertuples(index=False):
        rows.append({
            "Model": row.model,
            "AUROC": format_value(row.auroc, 4),
            "AUPRC": format_value(row.auprc, 4),
            "Accuracy": format_value(row.accuracy, 4),
            "Sensitivity": format_value(row.sensitivity, 4),
            "Specificity": format_value(row.specificity, 4),
            "F1": format_value(row.f1, 4),
            "Precision": format_value(row.precision, 4),
        })
    return rows


def cv_rows(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy()
    rows = []
    for row in df.itertuples(index=False):
        rows.append({
            "Model": row.model,
            "AUROC": f"{row.auroc_mean:.4f} +/- {row.auroc_std:.4f}",
            "AUPRC": f"{row.auprc_mean:.4f} +/- {row.auprc_std:.4f}",
            "Accuracy": f"{row.accuracy_mean:.4f} +/- {row.accuracy_std:.4f}",
            "Sensitivity": f"{row.sensitivity_mean:.4f} +/- {row.sensitivity_std:.4f}",
            "Specificity": f"{row.specificity_mean:.4f} +/- {row.specificity_std:.4f}",
            "F1": f"{row.f1_mean:.4f} +/- {row.f1_std:.4f}",
        })
    return rows


def ci_rows(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    metrics = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]
    rows = []
    for model, group in df.groupby("model", sort=False):
        row = {"Model": model}
        for metric in metrics:
            item = group[group["metric"] == metric].iloc[0]
            row[metric.upper()] = f"{item.estimate:.4f} [{item.ci_lower:.4f}, {item.ci_upper:.4f}]"
        rows.append(row)
    return rows


def ablation_status_rows() -> List[Dict[str, object]]:
    return [
        {"Model": "FETA-Transformer", "Ablation": "no continuous positional encoding", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "FETA-Transformer", "Ablation": "no attention pooling", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "FETA-Transformer", "Ablation": "no maternal cross-attention", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "PREG-Net", "Ablation": "no knowledge edges", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "PREG-Net", "Ablation": "random graph", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "PREG-Net", "Ablation": "no temporal edges", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
        {"Model": "PREG-Net", "Ablation": "no edge attention", "Status": "pending", "Reason": "Ablation switch not implemented/run"},
    ]


def hyperparameter_rows() -> List[Dict[str, object]]:
    rows = []
    run_dirs = {
        "PREG-Net": "results/preg_net",
        "FETA-Transformer": "results/feta_transformer",
        "Ensemble": "results/ensemble",
        "Learned Ensemble": "results/ensemble_learned",
    }
    for model, run_dir in run_dirs.items():
        config = read_json(Path(run_dir) / "run_config.json")
        args = config["args"]
        rows.append({
            "Model": model,
            "Epochs": args.get("epochs", ""),
            "Batch size": args.get("batch_size", ""),
            "Learning rate": args.get("lr", ""),
            "Weight decay": args.get("weight_decay", ""),
            "Patience": args.get("patience", ""),
            "Architecture": architecture_summary(model, args),
            "Fusion": "learned" if args.get("learned_fusion") else ("average" if model == "Ensemble" else ""),
        })
    rows.extend([
        {
            "Model": "Torch Logistic Regression / MLP",
            "Epochs": "300",
            "Batch size": "full batch",
            "Learning rate": "0.001",
            "Weight decay": "0.0001",
            "Patience": "50",
            "Architecture": "23 engineered tabular features; MLP hidden 64/32",
            "Fusion": "",
        },
        {
            "Model": "Sklearn / tree baselines",
            "Epochs": "",
            "Batch size": "",
            "Learning rate": "",
            "Weight decay": "",
            "Patience": "",
            "Architecture": "LogisticRegression, RandomForest, HistGradientBoosting, XGBoost on engineered tabular features",
            "Fusion": "",
        },
    ])
    return rows


def architecture_summary(model: str, args: Dict[str, object]) -> str:
    if model == "FETA-Transformer":
        return f"d_model={args.get('d_model')}, heads={args.get('heads')}, layers={args.get('layers')}, d_ff={args.get('d_ff')}, dropout={args.get('dropout')}"
    if model == "PREG-Net":
        return f"hidden_dim={args.get('hidden_dim')}, GAT layers={args.get('gat_layers')}, heads={args.get('heads')}, dropout={args.get('dropout')}"
    return "late fusion of trained FETA-Transformer and PREG-Net"


def parameter_count_from_checkpoint(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    return int(sum(tensor.numel() for tensor in state.values() if torch.is_tensor(tensor)))


def computational_cost_rows() -> List[Dict[str, object]]:
    rows = []
    run_dirs = {
        "PREG-Net": "results/preg_net",
        "FETA-Transformer": "results/feta_transformer",
        "Ensemble": "results/ensemble",
        "Learned Ensemble": "results/ensemble_learned",
    }
    for model, run_dir_str in run_dirs.items():
        run_dir = Path(run_dir_str)
        history = pd.read_csv(run_dir / "training_history.csv")
        checkpoint = run_dir / "checkpoints" / "best_model.pt"
        rows.append({
            "Model": model,
            "Parameters": str(parameter_count_from_checkpoint(checkpoint)),
            "Epochs completed": str(len(history)),
            "Total train time, s": f"{history['time_s'].sum():.1f}",
            "Mean epoch time, s": f"{history['time_s'].mean():.2f}",
            "Checkpoint size, MB": f"{checkpoint.stat().st_size / (1024 * 1024):.2f}",
            "Device": "CPU",
        })
    feature_count = len(read_json(Path("results/baselines/feature_columns.json")))
    rows.extend([
        {
            "Model": "Torch Logistic Regression",
            "Parameters": str(feature_count + 1),
            "Epochs completed": "not recorded",
            "Total train time, s": "not recorded",
            "Mean epoch time, s": "not recorded",
            "Checkpoint size, MB": "not persisted",
            "Device": "CPU",
        },
        {
            "Model": "Torch MLP",
            "Parameters": str(feature_count * 64 + 64 + 64 * 32 + 32 + 32 + 1),
            "Epochs completed": "not recorded",
            "Total train time, s": "not recorded",
            "Mean epoch time, s": "not recorded",
            "Checkpoint size, MB": "not persisted",
            "Device": "CPU",
        },
    ])
    return rows


def copy_case_svgs(explainability_dir: Path, figures_dir: Path) -> List[str]:
    copied = []
    for source in sorted((explainability_dir / "patient_graphs").glob("*.svg")):
        target = figures_dir / f"fig_graph_{source.name}"
        shutil.copyfile(source, target)
        copied.append(str(target))
    for source in sorted((explainability_dir / "patient_trajectories").glob("*.svg")):
        target = figures_dir / f"fig_trajectory_{source.name}"
        shutil.copyfile(source, target)
        copied.append(str(target))
    return copied


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    shutil.copyfile(source, target)
    return True


def write_summary(path: Path, figures: Sequence[str], tables: Sequence[str], ablation_real: bool) -> None:
    lines = [
        "# Paper Artifacts",
        "",
        "Paper-ready SVG figures and CSV/Markdown/LaTeX tables generated from saved experiment outputs.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- `{figure}`")
    lines.extend(["", "## Tables", ""])
    for table in tables:
        lines.append(f"- `{table}`")
    lines.extend([
        "",
        "## Ablation Note",
        "",
        "Real ablation experiments are not present in the saved result set." if not ablation_real else "Ablation artifacts were generated from real ablation outputs.",
        "The ablation status figure/table should not be interpreted as experimental evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(args.patients_csv)
    scans = pd.read_csv(args.scans_csv)
    comparison = pd.read_csv(args.comparison_csv)
    predictions = load_predictions()

    figure_paths: List[str] = []
    table_paths: List[str] = []

    write_architecture_svg(figures_dir / "fig_feta_architecture.svg", "FETA-Transformer")
    write_architecture_svg(figures_dir / "fig_preg_net_architecture.svg", "PREG-Net")
    write_roc_pr_figures(figures_dir, comparison, predictions)
    write_confusion_matrix_svg(figures_dir, predictions)

    explainability_dir = Path(args.explainability_dir)
    feta_week = pd.read_csv(explainability_dir / "feta_attention_by_gestational_week.csv")
    write_heatmap_svg(figures_dir / "fig_feta_temporal_attention_heatmap.svg", "FETA Temporal Attention By Gestational Week", feta_week, "mean_temporal_attention")
    write_heatmap_svg(figures_dir / "fig_feta_maternal_to_time_heatmap.svg", "Post-Hoc Maternal-To-Time Attention By Gestational Week", feta_week, "mean_maternal_to_temporal_attention")

    node_summary = pd.read_csv(explainability_dir / "preg_node_importance_summary.csv")
    edge_summary = pd.read_csv(explainability_dir / "preg_edge_attention_summary.csv")
    node_rows = [
        (f"{row.node_feature} ({row.node_family})", float(row.mean_importance))
        for row in node_summary.head(12).itertuples(index=False)
    ]
    edge_rows = [
        (str(row.relationship), float(row.mean_attention))
        for row in edge_summary.head(12).itertuples(index=False)
    ]
    write_bar_svg(figures_dir / "fig_preg_node_importance.svg", "PREG-Net Node Importance", node_rows, "Mean readout importance", "#2f5fb3")
    write_bar_svg(figures_dir / "fig_preg_edge_attention.svg", "PREG-Net Edge Attention", edge_rows, "Mean edge attention", "#c77736")

    if copy_if_exists(Path(args.calibration_svg), figures_dir / "fig_calibration_curves.svg"):
        pass
    copy_case_svgs(explainability_dir, figures_dir)
    write_ablation_status_svg(figures_dir / "fig_ablation_status_pending.svg")

    figure_paths = [str(path.relative_to(out_dir)) for path in sorted(figures_dir.glob("*.svg"))]

    table_specs = [
        ("table_dataset_demographics_by_outcome", build_demographics_table(patients, scans), "Dataset demographics by outcome", "tab:dataset-demographics"),
        ("table_model_comparison_metrics", rounded_model_comparison(Path(args.comparison_csv)), "Fixed test split model comparison", "tab:model-comparison"),
        ("table_cross_validation_summary", cv_rows(Path(args.cv_summary_csv)), "Five-fold cross-validation summary", "tab:cross-validation"),
        ("table_confidence_intervals", ci_rows(Path(args.ci_csv)), "Bootstrap confidence intervals", "tab:confidence-intervals"),
        ("table_ablation_status", ablation_status_rows(), "Ablation status", "tab:ablation-status"),
        ("table_hyperparameters", hyperparameter_rows(), "Model hyperparameters", "tab:hyperparameters"),
        ("table_computational_cost", computational_cost_rows(), "Computational cost", "tab:computational-cost"),
    ]
    for stem, rows, caption, label in table_specs:
        table_bundle(tables_dir, stem, rows, caption, label)
    table_paths = [str(path.relative_to(out_dir)) for path in sorted(tables_dir.glob("*"))]

    manifest = {
        "out_dir": str(out_dir),
        "figures": figure_paths,
        "tables": table_paths,
        "source_artifacts": {
            "model_comparison": args.comparison_csv,
            "cross_validation": args.cv_summary_csv,
            "confidence_intervals": args.ci_csv,
            "calibration": args.calibration_svg,
            "explainability": args.explainability_dir,
        },
        "ablation_note": "No real ablation result files were found; pending-status artifacts were generated only as placeholders.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(out_dir / "summary.md", figure_paths, table_paths, ablation_real=False)

    print(f"Generated paper artifacts in: {out_dir}")
    print(f"  Figures: {len(figure_paths)}")
    print(f"  Table files: {len(table_paths)}")
    print("  Ablation results: pending-status artifact only")


if __name__ == "__main__":
    main()
