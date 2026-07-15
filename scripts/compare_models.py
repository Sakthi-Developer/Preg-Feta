"""Train the two specialists and the gated hybrid, then print one metric table.

The iteration loop: change a model, re-run this, read the table.

    .venv/Scripts/python.exe scripts/compare_models.py

Trains on a single stratified split by default (fast). Pass --n-folds 5 for the
cross-validated numbers that go in the manuscript -- a single split is noisy, and
differences smaller than the fold-to-fold spread are not real.

Reuses cross_validate.py's plumbing, so a change to a model signature or to the
training loop is picked up here automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_validate import (
    build_loaders,
    make_folds,
    resolve_device,
    run_deep_model,
)

# model key -> (label for the table, what it sees when handicapped / on --full-data)
MODELS = {
    "feta": ("Model A  FETA-Transformer", "scans + maternal", "scans + maternal"),
    "preg": ("Model B  PREG-Net", "clinical only", "clinical + scans"),
    "hybrid": ("HYBRID   gated fusion", "both, gated", "both, gated"),
    "ensemble": ("(ref)    logit average", "both, averaged", "both, averaged"),
}
SHOWN = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients-csv", default="data/generated_dual/patients.csv")
    p.add_argument("--scans-csv", default="data/generated_dual/scans.csv")
    p.add_argument("--out-dir", default="results/compare")
    p.add_argument("--n-folds", type=int, default=1,
                   help="1 = single split (fast). 5 = cross-validated (report-grade).")
    p.add_argument("--with-ensemble", action="store_true",
                   help="Also train the plain logit-average ensemble. This is the "
                        "comparison that decides whether the gated architecture is "
                        "worth anything -- beating the specialists is guaranteed by "
                        "design, beating the ensemble is not.")
    p.add_argument("--full-data", action="store_true",
                   help="Un-handicap the specialists: PREG-Net's graph keeps its "
                        "ultrasound nodes (i.e. NOT --clinical-only), so every model "
                        "sees every modality. This is the ablation that decides whether "
                        "the fusion gain is architectural or just an artifact of "
                        "starving the specialists. (FETA always sees the maternal "
                        "features either way -- it has a maternal cross-attention "
                        "block -- so only PREG-Net and the hybrid change here.)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--verbose", action="store_true",
                   help="Show the Trainer's per-epoch lines. Off by default so the "
                        "progress table stays readable.")
    args = p.parse_args()
    # run_deep_model() reads these. clinical_only keeps the specialists
    # complementary (the paper's configuration); --full-data lifts that.
    args.clinical_only = not args.full_data
    args.hybrid_gated = True
    return args


def fmt(mean: float, std: float | None) -> str:
    return f"{mean:.3f}" if std is None else f"{mean:.3f} +/- {std:.3f}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    # FETA sets norm_first=True, which makes TransformerEncoder emit this on every
    # construction. Cosmetic; it would otherwise break up the progress table.
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

    device = resolve_device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(ROOT / args.patients_csv)
    scans = pd.read_csv(ROOT / args.scans_csv)

    models: List[str] = ["feta", "preg", "hybrid"]
    if args.with_ensemble:
        models.append("ensemble")

    prevalence = float(patients["label"].mean())
    cohort = Path(args.patients_csv).parent.name
    print(f"cohort     : {cohort}  ({len(patients):,} patients, "
          f"{prevalence:.1%} loss)")
    print(f"protocol   : {'single split' if args.n_folds == 1 else f'{args.n_folds}-fold CV'}"
          f"  |  device: {device}")
    print(f"regime     : {'FULL DATA -- every model sees every modality (ablation)'
                          if args.full_data else
                          'complementary specialists (--clinical-only; paper config)'}")
    # n_folds=1 is not a valid StratifiedKFold; use a 5-way split and keep one fold.
    folds = make_folds(patients, max(args.n_folds, 2), args.val_ratio, args.seed)
    if args.n_folds == 1:
        folds = folds[:1]

    total = len(folds) * len(models)
    print(f"jobs       : {total}  ({len(models)} models x {len(folds)} fold(s), "
          f"up to {args.epochs} epochs each)\n")
    print(f"{'':>5} {'job':<26} {'fold':<6} {'test AUROC':>10} {'mins':>6} {'ETA':>7}")
    print("-" * 65)

    # rows[model][metric] -> one value per fold
    rows: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in SHOWN} for m in models}
    done = 0
    t_start = time.time()
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
        loaders, _ = build_loaders(
            patients, scans, train_ids, val_ids, test_ids, args.batch_size
        )
        for name in models:
            t0 = time.time()
            # The Trainer logs per-epoch lines to the logger; this is the coarse
            # progress line, printed once each model finishes.
            split_metrics = run_deep_model(
                name, fold_idx, loaders, out_dir, args, device
            )
            for metric in SHOWN:
                rows[name][metric].append(float(split_metrics["test"][metric]))

            done += 1
            mins = (time.time() - t0) / 60.0
            eta = (time.time() - t_start) / done * (total - done) / 60.0
            print(f"{done:>2}/{total} {MODELS[name][0]:<26} {fold_idx + 1}/{len(folds):<4} "
                  f"{split_metrics['test']['auroc']:>10.3f} {mins:>6.1f} "
                  f"{eta:>6.1f}m", flush=True)
    print()

    multi = len(folds) > 1
    table = pd.DataFrame(
        [
            {
                "Model": MODELS[m][0],
                "Sees": MODELS[m][2 if args.full_data else 1],
                **{
                    k.upper() if k in ("auroc", "auprc") else k.capitalize():
                        fmt(float(np.mean(rows[m][k])),
                            float(np.std(rows[m][k], ddof=1)) if multi else None)
                    for k in SHOWN
                },
            }
            for m in models
        ]
    )

    print("=" * 100)
    print(f"TEST-SET METRICS{'  (mean +/- SD over folds)' if multi else '  (single split)'}")
    print("=" * 100)
    print(table.to_string(index=False))
    print("=" * 100)

    auroc = {m: float(np.mean(rows[m]["auroc"])) for m in models}
    best_spec = max(auroc["feta"], auroc["preg"])
    print(f"\nhybrid over best specialist : {auroc['hybrid'] - best_spec:+.3f}   "
          "(expected -- each specialist is denied half the data by design)")
    if "ensemble" in auroc:
        delta = auroc["hybrid"] - auroc["ensemble"]
        print(f"hybrid over plain ensemble  : {delta:+.3f}   "
              "<-- THE number: does gating beat averaging?")
        if multi:
            spread = float(np.std(rows["hybrid"]["auroc"], ddof=1))
            if abs(delta) < spread:
                print(f"    NOT significant: the gap is smaller than the hybrid's own "
                      f"fold-to-fold SD ({spread:.3f}).")
    else:
        print("hybrid over plain ensemble  : not measured -- re-run with --with-ensemble.")

    # Accuracy is near-meaningless at this prevalence; say so rather than let it be quoted.
    print(f"\nNOTE: predicting 'no loss' for everyone scores {1 - prevalence:.1%} accuracy "
          f"on this cohort.\n      Quote AUROC/AUPRC, not accuracy.")

    csv_path = out_dir / "comparison.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
