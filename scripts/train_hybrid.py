"""Train the hybrid cross-attention fusion of FETA-Transformer and PREG-Net.

Unlike the late-fusion ensemble (which only combines the two final logits), the
hybrid lets the temporal and graph towers attend to each other mid-network. An
auxiliary per-branch loss (``--aux-weight``) keeps both towers individually
predictive so neither collapses during joint training.
"""

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
from data.graph_builder import PatientGraphBuilder
from models import HybridFusionPredictor
from utils.metrics import compute_all_metrics
from utils.training import Trainer, compute_class_weights, get_best_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the FETA + PREG-Net hybrid cross-attention fusion model."
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
    parser.add_argument("--out-dir", default="results/hybrid")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-heads", type=int, default=4, help="Cross-attention heads.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.3,
        help="Auxiliary per-branch loss weight (anti-collapse). 0 disables it.",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        help="Only FETA attends to PREG (ablation); default is bidirectional.",
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help="Gated pure-branch fusion: aux heads on pure reps + per-patient gate "
             "routing (keeps towers complementary, avoids copycat collapse).",
    )
    parser.add_argument(
        "--clinical-only",
        action="store_true",
        help="Build the PREG-Net branch as a clinical specialist (maternal-only "
             "graph, no ultrasound nodes) so it complements the FETA ultrasound "
             "branch instead of overlapping. Use with warm-start checkpoints "
             "trained the same way.",
    )
    parser.add_argument(
        "--init-feta",
        default=None,
        help="Warm-start the FETA tower from a trained standalone checkpoint "
             "(e.g. results/feta_transformer/checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--init-preg",
        default=None,
        help="Warm-start the PREG-Net tower from a trained standalone checkpoint.",
    )
    parser.add_argument(
        "--freeze-towers",
        action="store_true",
        help="Freeze the warm-started towers; train only the fusion/gate. Branches "
             "then equal the strong, diverse standalones and the gate learns routing.",
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

    model = HybridFusionPredictor(
        n_heads=args.n_heads,
        dropout=args.dropout,
        bidirectional=not args.unidirectional,
        gated=args.gated,
    )

    # Optional warm-start of the towers from trained standalone checkpoints. The
    # standalone FETATransformer/PREGNet state dicts share parameter names with the
    # hybrid's self.feta / self.preg, so they load directly.
    def _load_tower(tower, ckpt_path, name):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = tower.load_state_dict(state, strict=False)
        print(f"Warm-started {name} from {ckpt_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    if args.init_feta:
        _load_tower(model.feta, args.init_feta, "FETA")
    if args.init_preg:
        _load_tower(model.preg, args.init_preg, "PREG-Net")
    if args.freeze_towers:
        for m in (model.feta, model.preg):
            for p in m.parameters():
                p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Froze towers; trainable params now {n_train:,} (fusion/gate only)")

    graph_builder = PatientGraphBuilder(
        max_scans=5, include_ultrasound=not args.clinical_only
    )
    if args.clinical_only:
        print("PREG-Net branch: clinical specialist (maternal-only graph, "
              f"{graph_builder.n_nodes} nodes)")

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
        aux_loss_weight=args.aux_weight,
        graph_builder=graph_builder,
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

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "norm_stats": {k: [float(v[0]), float(v[1])] for k, v in norm_stats.items()},
                "pos_weight": float(pos_weight),
                "best_auroc": float(trainer.best_auroc),
                "bidirectional": not args.unidirectional,
                "aux_weight": args.aux_weight,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    test_metrics = metrics_by_split["test"]
    print("\nHybrid test metrics")
    for key in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]:
        print(f"  {key}: {test_metrics[key]:.4f}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
