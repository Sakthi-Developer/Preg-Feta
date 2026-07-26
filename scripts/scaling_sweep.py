"""Measure the fusion gap as a function of dataset size (the n-scaling law).

THE QUESTION (pre-registered 2026-07-16, before any of this was run)
--------------------------------------------------------------------
`results/` already contains two points that disagree, on the SAME data-generating
process (`data/gen_mid_small` is literally the first 1200 patients of
`data/gen_dual_c05` -- labels agree 1.0000):

    n=1200, single split (results/probe_parity):  hybrid 0.971  ensemble 0.950  -> +0.021
    n=3000, 5-fold      (results/cv_c05_full):    hybrid 0.984  ensemble 0.986  -> -0.002

Note what moved: the hybrid barely improved (0.971 -> 0.984) while PREG-Net jumped
(0.927 -> 0.976). That is the signature of a baseline that had not yet converged, not
of a gate adding value.

    A real architectural advantage PERSISTS OR GROWS with n.
    A small-sample artifact SHRINKS like ~1/sqrt(n).

So the endpoint is the TREND, not any single cell:

    delta(n) = AUROC(hybrid) - max{ PREG-Net(full inputs), logit-average ensemble,
                                    HistGradientBoosting }

reported with fold SD at every n. EVERY CELL IS REPORTED regardless of outcome. Picking
the n where the hybrid happens to win would be selecting on noise -- which is exactly
what the two points above demonstrate.

Both outcomes are publishable:
  * delta -> 0 or negative as n grows: the fusion edge is a finite-sample artifact.
    An honest negative result, and a real methods contribution.
  * delta holds or grows: a genuine effect, and the CURVE is the evidence rather than
    one lucky configuration.

INPUT PARITY (CLAUDE.md rule): every model sees BOTH modalities. No --clinical-only.
Beating a handicapped specialist is guaranteed by construction and proves nothing.

BASELINES ARE NOT SANDBAGGED: every model gets the same folds, the same data, the same
epoch budget. The tree gets train+val (the same rows the deep models use for fitting and
early stopping). If the hybrid needs a weakened opponent to win, the result is fabricated.

COHORTS NEST: all four come from the b7ac269d generator at seed 42, corr 0.25,
coupling 0.5, so cohort-resampling variance does not contaminate the curve. (The one
exception is FHR imputation: the generator fills fully-missing FHR with the WHOLE-COHORT
median, which shifts 156.3 -> 156.5 between n=3000 and n=30000. That is 0.2 bpm against a
6.5 bpm measurement error -- negligible, but it is why the scan CSVs are not byte-identical
across sizes.)

Usage:
    .venv/Scripts/python.exe scripts/scaling_sweep.py --n-folds 5 --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
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
from utils.metrics import compute_all_metrics

DEEP_MODELS = ["feta", "preg", "hybrid", "ensemble"]
ALL_MODELS = DEEP_MODELS + ["gbt"]
LABELS = {
    "feta": "FETA-Transformer",
    "preg": "PREG-Net (full inputs)",
    "hybrid": "HYBRID gated fusion",
    "ensemble": "Ensemble (logit avg)",
    "gbt": "HistGradientBoosting",
}
# The hybrid must clear ALL of these on equal inputs, not just the two towers.
BASELINES = ["preg", "ensemble", "gbt"]
SHOWN = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]

DEFAULT_COHORTS = [
    "data/gen_mid_small",     # n=1200   (= first 1200 of gen_dual_c05)
    "data/gen_dual_c05",      # n=3000   (the current headline table)
    "data/gen_c05_n10000",    # n=10000
    "data/gen_c05_n30000",    # n=30000
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cohorts", nargs="+", default=DEFAULT_COHORTS,
                   help="Cohort directories, ascending in n. Must share a generator/seed.")
    p.add_argument("--out-dir", default="results/scaling")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    # INPUT PARITY: never handicap a baseline. See CLAUDE.md.
    args.clinical_only = False
    args.hybrid_gated = True
    return args


def build_feature_table_fast(patients: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    """Patient-level tabular features for the tree baseline.

    Same feature definitions as scripts/train_baselines.py, but grouped once instead of
    re-filtering the scan frame per patient -- that version is quadratic and costs ~16 min
    at n=30000, which is prohibitive here.
    """
    from scripts.train_baselines import MATERNAL_FEATURES, US_FEATURES

    patients = patients.copy()
    patients["conception_ivf"] = (
        patients["conception"].astype(str).str.upper() == "IVF"
    ).astype(float)
    patients["singleton"] = patients["singleton"].astype(float)

    scans = scans.sort_values(["patient_id", "gestational_age_weeks"])
    g = scans.groupby("patient_id", sort=False)

    out = pd.DataFrame(index=g.size().index)
    out["scan_count"] = g.size().astype(float)
    ga = g["gestational_age_weeks"]
    out["ga_first"] = ga.first()
    out["ga_latest"] = ga.last()
    out["ga_span"] = out["ga_latest"] - out["ga_first"]

    for feat in US_FEATURES:
        out[f"{feat}_latest"] = g[feat].last()
        out[f"{feat}_mean"] = g[feat].mean()
        # OLS slope over GA, vectorised: cov(x,y)/var(x); 0 when x is degenerate.
        x = scans["gestational_age_weeks"]
        y = scans[feat]
        xm = x.groupby(scans["patient_id"], sort=False).transform("mean")
        ym = y.groupby(scans["patient_id"], sort=False).transform("mean")
        num = ((x - xm) * (y - ym)).groupby(scans["patient_id"], sort=False).sum()
        den = ((x - xm) ** 2).groupby(scans["patient_id"], sort=False).sum()
        out[f"{feat}_slope"] = np.where(den > 1e-12, num / den.replace(0, np.nan), 0.0)

    out = out.reset_index()
    keep = ["patient_id", "label"] + MATERNAL_FEATURES
    merged = out.merge(patients[keep], on="patient_id", how="left")
    return merged


def run_gbt(feat: pd.DataFrame, train_ids, val_ids, test_ids) -> Dict[str, float]:
    """HistGradientBoosting on the same fold. Gets train+val, like the deep models."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    drop = ["patient_id", "label"]
    fit_ids = list(train_ids) + list(val_ids)
    tr = feat[feat["patient_id"].isin(fit_ids)]
    te = feat[feat["patient_id"].isin(test_ids)]
    X_tr = tr.drop(columns=drop).to_numpy(dtype=float)
    y_tr = tr["label"].to_numpy(dtype=int)
    X_te = te.drop(columns=drop).to_numpy(dtype=float)
    y_te = te["label"].to_numpy(dtype=int)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, random_state=42
    )
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return compute_all_metrics(y_te, prob)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

    device = resolve_device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device : {device}   |   {args.n_folds}-fold CV   |   INPUT PARITY "
          f"(every model sees both modalities)")
    print(f"endpoint : delta = hybrid - max({', '.join(LABELS[b] for b in BASELINES)})")
    print("all cells reported; the trend in delta vs n is the result.\n")

    records: List[Dict] = []
    t_start = time.time()

    for cohort in args.cohorts:
        cdir = ROOT / cohort
        patients = pd.read_csv(cdir / "patients.csv")
        scans = pd.read_csv(cdir / "scans.csv")
        n = len(patients)
        print(f"=== {cohort}  (n={n:,}, {patients['label'].mean():.1%} loss) ===",
              flush=True)

        t0 = time.time()
        feat = build_feature_table_fast(patients, scans)
        print(f"    tree features: {feat.shape[0]}x{feat.shape[1] - 2} "
              f"({time.time() - t0:.0f}s)", flush=True)

        folds = make_folds(patients, args.n_folds, args.val_ratio, args.seed)
        for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
            loaders, _ = build_loaders(
                patients, scans, train_ids, val_ids, test_ids, args.batch_size
            )
            for name in DEEP_MODELS:
                t1 = time.time()
                m = run_deep_model(
                    name, fold_idx, loaders, out_dir / f"n{n}", args, device
                )["test"]
                records.append({"n": n, "model": name, "fold": fold_idx,
                                **{k: float(m[k]) for k in SHOWN}})
                print(f"    n={n:<6} fold {fold_idx + 1}/{len(folds)}  "
                      f"{LABELS[name]:<22} AUROC {m['auroc']:.3f}  "
                      f"({(time.time() - t1) / 60:.1f}m)", flush=True)

            m = run_gbt(feat, train_ids, val_ids, test_ids)
            records.append({"n": n, "model": "gbt", "fold": fold_idx,
                            **{k: float(m[k]) for k in SHOWN}})
            print(f"    n={n:<6} fold {fold_idx + 1}/{len(folds)}  "
                  f"{LABELS['gbt']:<22} AUROC {m['auroc']:.3f}", flush=True)

            # Checkpoint after every fold -- this sweep runs for hours.
            pd.DataFrame(records).to_csv(out_dir / "raw_folds.csv", index=False)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "raw_folds.csv", index=False)

    agg = (df.groupby(["n", "model"])["auroc"]
             .agg(mean="mean", sd=lambda s: s.std(ddof=1))
             .reset_index())
    agg.to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 78)
    print("AUROC by dataset size (mean +/- SD over folds)")
    print("=" * 78)
    piv = agg.pivot(index="model", columns="n", values="mean")
    sd = agg.pivot(index="model", columns="n", values="sd")
    for m in ALL_MODELS:
        if m not in piv.index:
            continue
        cells = "  ".join(f"{piv.loc[m, c]:.3f}+/-{sd.loc[m, c]:.3f}" for c in piv.columns)
        print(f"{LABELS[m]:<24} {cells}")

    print("\n" + "=" * 78)
    print("THE RESULT: delta = hybrid - best same-input baseline")
    print("=" * 78)
    print(f"{'n':<9} {'hybrid':>7} {'best baseline':>26} {'delta':>8} {'hybrid SD':>10}  verdict")
    for c in piv.columns:
        h = piv.loc["hybrid", c]
        bests = {b: piv.loc[b, c] for b in BASELINES if b in piv.index}
        bname = max(bests, key=bests.get)
        d = h - bests[bname]
        hsd = sd.loc["hybrid", c]
        verdict = ("WITHIN NOISE" if abs(d) < hsd else
                   ("hybrid ahead" if d > 0 else "hybrid BEHIND"))
        print(f"{c:<9} {h:>7.3f} {LABELS[bname] + f' {bests[bname]:.3f}':>26} "
              f"{d:>+8.3f} {hsd:>10.3f}  {verdict}")

    ns = list(piv.columns)
    deltas = [piv.loc["hybrid", c] - max(piv.loc[b, c] for b in BASELINES if b in piv.index)
              for c in ns]
    print(f"\ntrend: delta goes {' -> '.join(f'{d:+.3f}' for d in deltas)} "
          f"as n goes {' -> '.join(str(c) for c in ns)}")
    if len(ns) > 1:
        if deltas[-1] < deltas[0]:
            print("  delta SHRINKS with n -> consistent with a finite-sample artifact,")
            print("  NOT with an architectural advantage. Report this honestly.")
        else:
            print("  delta HOLDS/GROWS with n -> consistent with a real effect.")
            print("  Confirm it clears the fold SD before claiming it.")

    print(f"\ntotal: {(time.time() - t_start) / 60:.0f} min")
    print(f"wrote {(out_dir / 'summary.csv').relative_to(ROOT)} and raw_folds.csv")


if __name__ == "__main__":
    main()
