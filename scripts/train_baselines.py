"""Train tabular baseline models on patient-level engineered features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import create_splits
from utils.metrics import compute_all_metrics


US_FEATURES = ["FHR", "CRL", "GS", "YSD"]
MATERNAL_FEATURES = [
    "age",
    "bmi",
    "parity",
    "gravidity",
    "previous_loss",
    "conception_ivf",
    "singleton",
]


class TorchLogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TorchMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular baseline models.")
    parser.add_argument("--patients-csv", default="data/generated/patients.csv")
    parser.add_argument("--scans-csv", default="data/generated/scans.csv")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return parser.parse_args()


def slope_over_ga(group: pd.DataFrame, feature: str) -> float:
    values = group[["gestational_age_weeks", feature]].dropna()
    if len(values) < 2:
        return 0.0
    x = values["gestational_age_weeks"].to_numpy(dtype=float)
    y = values[feature].to_numpy(dtype=float)
    if np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, deg=1)[0])


def build_feature_table(patients: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    patients = patients.copy()
    patients["conception_ivf"] = (
        patients["conception"].astype(str).str.upper() == "IVF"
    ).astype(float)
    patients["singleton"] = patients["singleton"].astype(float)

    rows: List[Dict[str, float | str | int]] = []
    for _, patient in patients.iterrows():
        pid = patient["patient_id"]
        group = scans[scans["patient_id"] == pid].sort_values("gestational_age_weeks")
        row: Dict[str, float | str | int] = {
            "patient_id": pid,
            "label": int(patient["label"]),
            "scan_count": float(len(group)),
            "ga_first": float(group["gestational_age_weeks"].iloc[0]),
            "ga_latest": float(group["gestational_age_weeks"].iloc[-1]),
            "ga_span": float(
                group["gestational_age_weeks"].iloc[-1]
                - group["gestational_age_weeks"].iloc[0]
            ),
        }

        for feature in MATERNAL_FEATURES:
            row[feature] = float(patient[feature])

        for feature in US_FEATURES:
            row[f"{feature}_latest"] = float(group[feature].iloc[-1])
            row[f"{feature}_mean"] = float(group[feature].mean())
            row[f"{feature}_slope"] = slope_over_ga(group, feature)

        rows.append(row)

    return pd.DataFrame(rows)


def split_and_normalize(
    features: pd.DataFrame,
    seed: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str], Dict[str, Dict[str, float]]]:
    train_ids, val_ids, test_ids = create_splits(features, seed=seed)
    feature_cols = [
        col for col in features.columns if col not in {"patient_id", "label"}
    ]

    train_df = features[features["patient_id"].isin(train_ids)]
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)

    stats = {
        col: {"mean": float(means[col]), "std": float(stds[col])}
        for col in feature_cols
    }

    splits = {}
    for split_name, ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        df = features[features["patient_id"].isin(ids)].copy()
        x = ((df[feature_cols] - means) / stds).to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.float32)
        splits[split_name] = {
            "patient_id": df["patient_id"].to_numpy(),
            "x": x,
            "y": y,
        }

    return splits, feature_cols, stats


def pos_weight(y: np.ndarray) -> float:
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    return n_neg / max(n_pos, 1.0)


def train_torch_model(
    name: str,
    model: nn.Module,
    splits: Dict[str, Dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight(splits["train"]["y"])], device=device)
    )

    x_train = torch.tensor(splits["train"]["x"], device=device)
    y_train = torch.tensor(splits["train"]["y"], device=device)
    x_val = torch.tensor(splits["val"]["x"], device=device)
    y_val = splits["val"]["y"]

    best_state = deepcopy(model.state_dict())
    best_auroc = -math.inf
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
        metrics = compute_all_metrics(y_val, val_prob)
        val_auroc = metrics["auroc"]

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    predictions = predict_torch_model(model, splits, device)
    metrics_by_split = {
        split: compute_all_metrics(splits[split]["y"], predictions[split])
        for split in ["train", "val", "test"]
    }
    print(
        f"{name}: best_val_auroc={best_auroc:.4f}, "
        f"test_auroc={metrics_by_split['test']['auroc']:.4f}, "
        f"test_f1={metrics_by_split['test']['f1']:.4f}"
    )
    return metrics_by_split, predictions


@torch.no_grad()
def predict_torch_model(
    model: nn.Module,
    splits: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    predictions = {}
    for split in ["train", "val", "test"]:
        x = torch.tensor(splits[split]["x"], device=device)
        predictions[split] = torch.sigmoid(model(x)).detach().cpu().numpy()
    return predictions


def train_optional_sklearn(
    splits: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]]:
    results = {}
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        print("scikit-learn unavailable; skipping sklearn baselines.")
        return results

    models = {
        "sklearn_logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        print("xgboost unavailable; skipping XGBoost baseline.")

    sample_weight = np.where(
        splits["train"]["y"] == 1,
        pos_weight(splits["train"]["y"]),
        1.0,
    )
    for name, model in models.items():
        try:
            if name in {"hist_gradient_boosting", "xgboost"}:
                model.fit(splits["train"]["x"], splits["train"]["y"], sample_weight=sample_weight)
            else:
                model.fit(splits["train"]["x"], splits["train"]["y"])

            predictions = {}
            for split in ["train", "val", "test"]:
                predictions[split] = model.predict_proba(splits[split]["x"])[:, 1]

            metrics = {
                split: compute_all_metrics(splits[split]["y"], predictions[split])
                for split in ["train", "val", "test"]
            }
            print(
                f"{name}: test_auroc={metrics['test']['auroc']:.4f}, "
                f"test_f1={metrics['test']['f1']:.4f}"
            )
            results[name] = (metrics, predictions)
        except Exception as exc:
            print(f"{name} failed: {type(exc).__name__}: {exc}")

    return results


def save_predictions(
    out_dir: Path,
    model_name: str,
    splits: Dict[str, Dict[str, np.ndarray]],
    predictions: Dict[str, np.ndarray],
) -> None:
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        with (model_dir / f"{split}_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["patient_id", "label", "probability", "prediction"],
            )
            writer.writeheader()
            for pid, label, prob in zip(
                splits[split]["patient_id"],
                splits[split]["y"],
                predictions[split],
            ):
                writer.writerow({
                    "patient_id": pid,
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(prob >= 0.5),
                })


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(args.patients_csv)
    scans = pd.read_csv(args.scans_csv)
    features = build_feature_table(patients, scans)
    splits, feature_cols, stats = split_and_normalize(features, args.seed)

    print(
        f"Feature table: {len(features)} patients, "
        f"{len(feature_cols)} tabular features"
    )

    n_features = len(feature_cols)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    torch_models = {
        "torch_logistic_regression": TorchLogisticRegression(n_features),
        "torch_mlp": TorchMLP(n_features),
    }
    for name, model in torch_models.items():
        metrics, predictions = train_torch_model(name, model, splits, args)
        results[name] = metrics
        save_predictions(out_dir, name, splits, predictions)

    for name, (metrics, predictions) in train_optional_sklearn(splits).items():
        results[name] = metrics
        save_predictions(out_dir, name, splits, predictions)

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (out_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2),
        encoding="utf-8",
    )
    (out_dir / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(f"Saved baseline outputs to: {out_dir}")


if __name__ == "__main__":
    main()
