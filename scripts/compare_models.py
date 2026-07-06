"""Compare saved model metrics across trained runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_RUNS = {
    "PREG-Net": "results/preg_net",
    "FETA-Transformer": "results/feta_transformer",
    "Ensemble": "results/ensemble",
    "Learned Ensemble": "results/ensemble_learned",
}

BASELINE_LABELS = {
    "torch_logistic_regression": "Logistic Regression",
    "torch_mlp": "MLP",
    "sklearn_logistic_regression": "Sklearn Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
}

METRIC_ORDER = [
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained model metrics.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-prefix", default="results/model_comparison")
    return parser.parse_args()


def load_rows(split: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, run_dir in DEFAULT_RUNS.items():
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        split_metrics = metrics[split]
        row: Dict[str, object] = {"model": name, "run_dir": run_dir}
        for metric in METRIC_ORDER:
            row[metric] = float(split_metrics[metric])
        rows.append(row)

    baseline_path = Path("results/baselines/metrics.json")
    if baseline_path.exists():
        baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        for key, metrics in baseline_metrics.items():
            row = {
                "model": BASELINE_LABELS.get(key, key),
                "run_dir": f"results/baselines/{key}",
            }
            split_metrics = metrics[split]
            for metric in METRIC_ORDER:
                row[metric] = float(split_metrics[metric])
            rows.append(row)

    rows.sort(key=lambda row: row["auroc"], reverse=True)
    return rows


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "run_dir", *METRIC_ORDER])
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: List[Dict[str, object]], path: Path, split: str) -> None:
    headers = ["Model", *[metric.upper() for metric in METRIC_ORDER]]
    lines = [
        f"# Model Comparison ({split})",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row["model"])]
        values.extend(f"{row[metric]:.4f}" for metric in METRIC_ORDER)
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: List[Dict[str, object]], split: str) -> None:
    print(f"\nModel comparison ({split})")
    print("-" * 92)
    print(
        f"{'Model':<20} {'AUROC':>7} {'AUPRC':>7} {'Acc':>7} "
        f"{'Sens':>7} {'Spec':>7} {'F1':>7}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{row['model']:<20} {row['auroc']:>7.4f} {row['auprc']:>7.4f} "
            f"{row['accuracy']:>7.4f} {row['sensitivity']:>7.4f} "
            f"{row['specificity']:>7.4f} {row['f1']:>7.4f}"
        )
    print("-" * 92)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.split)
    if not rows:
        raise SystemExit("No metrics files found.")

    out_prefix = Path(args.out_prefix)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")

    save_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_markdown(rows, md_path, args.split)
    print_table(rows, args.split)
    print(f"Saved: {csv_path}, {json_path}, {md_path}")


if __name__ == "__main__":
    main()
