"""Train the late-fusion ensemble of FETA-Transformer and PREG-Net."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
from models import EnsemblePredictor
from utils.metrics import compute_all_metrics
from utils.training import Trainer, compute_class_weights, get_best_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the FETA + PREG-Net late-fusion ensemble."
    )
    parser.add_argument(
        "--patients-csv",
        default="data/generated/patients.csv",
        help="Path to generated patients CSV.",
    )
    parser.add_argument(
        "--scans-csv",
        default="data/generated/scans.csv",
        help="Path to generated scans CSV.",
    )
    parser.add_argument("--out-dir", default="results/ensemble")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--learned-fusion",
        action="store_true",
        help="Learn a 2-to-1 fusion layer instead of simple averaging.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "auto", "cuda", "mps"],
        default="cpu",
        help="Use cpu by default for widest Mac compatibility.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_best_device()
    return torch.device(name)


def save_history(history: List[Dict], out_dir: Path) -> None:
    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    if not history:
        return

    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def evaluate_loader(trainer: Trainer, loader, split_name: str, out_dir: Path) -> Dict[str, float]:
    trainer.model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []

    for batch in loader:
        patient_ids = list(batch.get("patient_id", []))
        inputs, labels = trainer._unpack_batch(batch)
        outputs = trainer.model(**inputs)

        probs = torch.sigmoid(outputs["logits"].squeeze(-1)).detach().cpu().numpy()
        feta_probs = torch.sigmoid(outputs["feta_logits"].squeeze(-1)).detach().cpu().numpy()
        preg_probs = torch.sigmoid(outputs["preg_logits"].squeeze(-1)).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels_np)

        for pid, label, prob, feta_prob, preg_prob in zip(
            patient_ids, labels_np, probs, feta_probs, preg_probs
        ):
            rows.append({
                "patient_id": pid,
                "label": int(label),
                "probability": float(prob),
                "prediction": int(prob >= 0.5),
                "feta_probability": float(feta_prob),
                "preg_probability": float(preg_prob),
            })

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    metrics = compute_all_metrics(y_true, y_prob)

    with (out_dir / f"{split_name}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "label",
                "probability",
                "prediction",
                "feta_probability",
                "preg_probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    patients_csv = Path(args.patients_csv)
    scans_csv = Path(args.scans_csv)
    if not patients_csv.exists() or not scans_csv.exists():
        raise FileNotFoundError(
            "Generated CSVs not found. Run: "
            "python3 synthetic_generator_v2.py --out data/generated --seed 42 --n-patients 800"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    loaders, norm_stats = get_dataloaders(
        str(patients_csv),
        str(scans_csv),
        batch_size=args.batch_size,
        max_scans=5,
        seed=args.seed,
    )

    train_labels = loaders["train"].dataset.patients["label"].to_numpy()
    pos_weight = compute_class_weights(train_labels)
    print(f"Computed pos_weight: {pos_weight:.3f}")

    model = EnsemblePredictor(learned_fusion=args.learned_fusion)
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        collate_mode="ensemble",
    )

    history = trainer.fit(args.epochs)
    save_history(history, out_dir)

    best_checkpoint = checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    metrics_by_split = {
        split: evaluate_loader(trainer, loader, split, out_dir)
        for split, loader in loaders.items()
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2),
        encoding="utf-8",
    )

    fusion_weights = None
    if args.learned_fusion and hasattr(trainer.model, "fusion_layer"):
        fusion_weights = (
            trainer.model.fusion_layer.weight.detach().cpu().squeeze().tolist()
        )

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "norm_stats": {k: [float(v[0]), float(v[1])] for k, v in norm_stats.items()},
                "pos_weight": float(pos_weight),
                "best_auroc": float(trainer.best_auroc),
                "fusion_weights": fusion_weights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    test_metrics = metrics_by_split["test"]
    print("\nEnsemble test metrics")
    for key in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]:
        print(f"  {key}: {test_metrics[key]:.4f}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
