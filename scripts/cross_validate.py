"""Stratified cross-validation for research-grade model reporting."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PregnancyDataset, compute_norm_stats
from models import EnsemblePredictor, FETATransformer, PREGNet
from scripts.train_baselines import (
    TorchLogisticRegression,
    TorchMLP,
    build_feature_table,
    train_optional_sklearn,
    train_torch_model,
)
from utils.metrics import compute_all_metrics
from utils.training import Trainer, compute_class_weights, get_best_device

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "cross_validate.py requires scikit-learn. Install requirements first."
    ) from exc


DEEP_MODELS = {"preg", "feta", "ensemble", "learned_ensemble"}
TORCH_BASELINES = {"torch_logistic_regression", "torch_mlp"}
SKLEARN_BASELINES = {
    "sklearn_logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
}
ALL_MODELS = DEEP_MODELS | TORCH_BASELINES | SKLEARN_BASELINES
METRICS = [
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
    "recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified cross-validation.")
    parser.add_argument("--patients-csv", default="data/generated/patients.csv")
    parser.add_argument("--scans-csv", default="data/generated/scans.csv")
    parser.add_argument("--out-dir", default="results/cross_validation")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--models",
        default="preg,feta,ensemble,learned_ensemble,torch_mlp,sklearn_logistic_regression,random_forest,hist_gradient_boosting,xgboost",
        help="Comma-separated model names or 'all'.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--device",
        choices=["cpu", "auto", "cuda", "mps"],
        default="cpu",
        help="Use cpu by default for widest Mac compatibility.",
    )
    return parser.parse_args()


def parse_models(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return sorted(ALL_MODELS)
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Valid: {sorted(ALL_MODELS)}")
    return models


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_best_device()
    return torch.device(name)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "patient_id": [b["patient_id"] for b in batch],
        "temporal_features": torch.stack([b["temporal_features"] for b in batch]),
        "gestational_ages": torch.stack([b["gestational_ages"] for b in batch]),
        "temporal_mask": torch.stack([b["temporal_mask"] for b in batch]),
        "maternal_features": torch.stack([b["maternal_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def make_folds(
    patients: pd.DataFrame,
    n_folds: int,
    val_ratio: float,
    seed: int,
) -> List[Tuple[List[str], List[str], List[str]]]:
    ids = patients["patient_id"].to_numpy(dtype=str)
    labels = patients["label"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        train_val_ids = ids[train_val_idx]
        train_val_labels = labels[train_val_idx]
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio,
            random_state=seed + fold_idx,
            stratify=train_val_labels,
        )
        test_ids = ids[test_idx]
        folds.append((train_ids.tolist(), val_ids.tolist(), test_ids.tolist()))

    return folds


def build_loaders(
    patients: pd.DataFrame,
    scans: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, Tuple[float, float]]]:
    norm_stats = compute_norm_stats(patients, scans, train_ids)
    loaders = {}
    for split, ids, shuffle in [
        ("train", train_ids, True),
        ("val", val_ids, False),
        ("test", test_ids, False),
    ]:
        split_patients = patients[patients["patient_id"].isin(ids)]
        split_scans = scans[scans["patient_id"].isin(ids)]
        dataset = PregnancyDataset(
            patients_df=split_patients,
            scans_df=split_scans,
            max_scans=5,
            norm_stats=norm_stats,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        )

    return loaders, norm_stats


def build_deep_model(model_name: str):
    if model_name == "preg":
        return PREGNet(), "preg"
    if model_name == "feta":
        return FETATransformer(), "feta"
    if model_name == "ensemble":
        return EnsemblePredictor(), "ensemble"
    if model_name == "learned_ensemble":
        return EnsemblePredictor(learned_fusion=True), "ensemble"
    raise ValueError(f"Not a deep model: {model_name}")


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


@torch.no_grad()
def evaluate_deep_model(trainer: Trainer, loader) -> Dict[str, float]:
    trainer.model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        inputs, labels = trainer._unpack_batch(batch)
        outputs = trainer.model(**inputs)
        logits = extract_logits(outputs).squeeze(-1)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    return compute_all_metrics(np.concatenate(all_labels), np.concatenate(all_probs))


def run_deep_model(
    model_name: str,
    fold_idx: int,
    loaders: Dict[str, DataLoader],
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model, collate_mode = build_deep_model(model_name)
    train_labels = loaders["train"].dataset.patients["label"].to_numpy()
    checkpoint_dir = out_dir / model_name / f"fold_{fold_idx}" / "checkpoints"

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=compute_class_weights(train_labels),
        patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        collate_mode=collate_mode,
    )
    trainer.fit(args.epochs)

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    return {
        split: evaluate_deep_model(trainer, loader)
        for split, loader in loaders.items()
    }


def make_tabular_splits(
    features: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    feature_cols = [
        col for col in features.columns if col not in {"patient_id", "label"}
    ]
    train_df = features[features["patient_id"].isin(train_ids)]
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)

    splits = {}
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        df = features[features["patient_id"].isin(ids)].copy()
        splits[split] = {
            "patient_id": df["patient_id"].to_numpy(),
            "x": ((df[feature_cols] - means) / stds).to_numpy(dtype=np.float32),
            "y": df["label"].to_numpy(dtype=np.float32),
        }
    return splits, feature_cols


def run_baselines(
    selected_models: Iterable[str],
    fold_idx: int,
    tabular_splits: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    selected = set(selected_models)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    if "torch_logistic_regression" in selected:
        n_features = tabular_splits["train"]["x"].shape[1]
        metrics, _ = train_torch_model(
            f"cv_fold_{fold_idx}_torch_logistic_regression",
            TorchLogisticRegression(n_features),
            tabular_splits,
            args,
        )
        results["torch_logistic_regression"] = metrics

    if "torch_mlp" in selected:
        n_features = tabular_splits["train"]["x"].shape[1]
        metrics, _ = train_torch_model(
            f"cv_fold_{fold_idx}_torch_mlp",
            TorchMLP(n_features),
            tabular_splits,
            args,
        )
        results["torch_mlp"] = metrics

    sklearn_requested = selected & SKLEARN_BASELINES
    if sklearn_requested:
        sklearn_results = train_optional_sklearn(tabular_splits)
        for model_name in sklearn_requested:
            if model_name in sklearn_results:
                results[model_name] = sklearn_results[model_name][0]

    return results


def append_metric_rows(
    rows: List[Dict[str, object]],
    model_name: str,
    fold_idx: int,
    split_metrics: Dict[str, Dict[str, float]],
) -> None:
    for split, metrics in split_metrics.items():
        row = {"model": model_name, "fold": fold_idx, "split": split}
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)


def save_fold_metrics(rows: List[Dict[str, object]], out_dir: Path) -> None:
    path = out_dir / "fold_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fold", "split", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "fold_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )


def save_fold_split(
    out_dir: Path,
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / f"fold_{fold_idx}.json").write_text(
        json.dumps(
            {
                "fold": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(rows: List[Dict[str, object]], out_dir: Path) -> None:
    summary_rows = []
    df = pd.DataFrame(rows)
    for (model, split), group in df.groupby(["model", "split"]):
        row = {"model": model, "split": split, "n_folds": int(len(group))}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: (row["split"] != "test", -row["auroc_mean"], row["model"])
    )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "split", "n_folds"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    test_rows = [row for row in summary_rows if row["split"] == "test"]
    lines = [
        "# Cross-Validation Summary (test)",
        "",
        "| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in test_rows:
        values = [
            row["model"],
            f"{row['auroc_mean']:.4f} +/- {row['auroc_std']:.4f}",
            f"{row['auprc_mean']:.4f} +/- {row['auprc_std']:.4f}",
            f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}",
            f"{row['sensitivity_mean']:.4f} +/- {row['sensitivity_std']:.4f}",
            f"{row['specificity_mean']:.4f} +/- {row['specificity_std']:.4f}",
            f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    selected_models = parse_models(args.models)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(args.patients_csv)
    scans = pd.read_csv(args.scans_csv)
    features = build_feature_table(patients, scans)
    folds = make_folds(patients, args.n_folds, args.val_ratio, args.seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Running {len(folds)} fold(s) on device: {device}")

    rows: List[Dict[str, object]] = []
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
        print(
            f"\nFold {fold_idx}: train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}"
        )
        save_fold_split(out_dir, fold_idx, train_ids, val_ids, test_ids)
        loaders, _ = build_loaders(
            patients, scans, train_ids, val_ids, test_ids, args.batch_size
        )
        tabular_splits, _ = make_tabular_splits(features, train_ids, val_ids, test_ids)

        for model_name in selected_models:
            if model_name in DEEP_MODELS:
                print(f"Training {model_name}...")
                metrics = run_deep_model(
                    model_name, fold_idx, loaders, out_dir, args, device
                )
                append_metric_rows(rows, model_name, fold_idx, metrics)

        baseline_models = [model for model in selected_models if model not in DEEP_MODELS]
        if baseline_models:
            baseline_results = run_baselines(
                baseline_models, fold_idx, tabular_splits, out_dir, args
            )
            for model_name, metrics in baseline_results.items():
                append_metric_rows(rows, model_name, fold_idx, metrics)

        save_fold_metrics(rows, out_dir)
        summarize(rows, out_dir)

    save_fold_metrics(rows, out_dir)
    summarize(rows, out_dir)
    print(f"\nSaved cross-validation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
