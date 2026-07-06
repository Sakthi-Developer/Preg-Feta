"""Single-batch overfit check for FETA, PREG-Net, and ensemble models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
from models import EnsemblePredictor, FETATransformer, PREGNet
from utils.training import Trainer, get_best_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that each model can overfit one real data batch."
    )
    parser.add_argument(
        "--mode",
        choices=["feta", "preg", "ensemble", "all"],
        default="all",
        help="Model path to test.",
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
    parser.add_argument("--steps", type=int, default=80, help="Optimization steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.05,
        help="Minimum relative loss drop required to pass.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when MPS/CUDA is available.",
    )
    return parser.parse_args()


def pick_mixed_batch(loader) -> Dict[str, torch.Tensor]:
    """Prefer a batch containing both classes for a stronger overfit check."""
    fallback = None
    for batch in loader:
        fallback = fallback or batch
        labels = batch["label"]
        if labels.sum() > 0 and labels.sum() < labels.numel():
            return batch
    return fallback


def build_model(mode: str) -> nn.Module:
    if mode == "feta":
        return FETATransformer()
    if mode == "preg":
        return PREGNet()
    if mode == "ensemble":
        return EnsemblePredictor()
    raise ValueError(f"Unsupported mode: {mode}")


def get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def run_mode(
    mode: str,
    batch: Dict[str, torch.Tensor],
    loader,
    device: torch.device,
    steps: int,
    lr: float,
    min_drop: float,
) -> Tuple[float, float, bool]:
    model = build_model(mode)
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        collate_mode=mode,
        checkpoint_dir=f"tmp/overfit_{mode}",
        device=device,
    )
    model = trainer.model
    inputs, labels = trainer._unpack_batch(batch)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()

    first_loss = None
    last_loss = None

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = get_logits(model(**inputs)).squeeze(-1)
        loss = criterion(logits, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value

        if step == 1 or step % 20 == 0 or step == steps:
            print(f"{mode:8s} step {step:03d}/{steps}: loss={value:.4f}")

    relative_drop = (first_loss - last_loss) / max(first_loss, 1e-8)
    passed = relative_drop >= min_drop
    status = "PASS" if passed else "FAIL"
    print(
        f"{mode:8s} {status}: initial={first_loss:.4f}, "
        f"final={last_loss:.4f}, drop={100 * relative_drop:.1f}%"
    )
    return first_loss, last_loss, passed


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu else get_best_device()
    print(f"Using device: {device}")

    loaders, _ = get_dataloaders(
        args.patients_csv,
        args.scans_csv,
        batch_size=args.batch_size,
        max_scans=5,
        seed=42,
    )
    batch = pick_mixed_batch(loaders["train"])
    if batch is None:
        raise RuntimeError("No training batch found.")

    labels = batch["label"]
    print(
        f"Selected batch: n={labels.numel()}, positives={int(labels.sum())}, "
        f"negatives={int(labels.numel() - labels.sum())}"
    )

    modes = ["feta", "preg", "ensemble"] if args.mode == "all" else [args.mode]
    results = [
        run_mode(mode, batch, loaders["train"], device, args.steps, args.lr, args.min_drop)
        for mode in modes
    ]

    if not all(passed for _, _, passed in results):
        raise SystemExit(1)

    print("Overfit check passed.")


if __name__ == "__main__":
    main()
