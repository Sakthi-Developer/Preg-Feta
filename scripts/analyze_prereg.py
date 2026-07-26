"""Apply the PREREGISTRATION.md decision rule to a compare_models.py run.

Reads per-fold `test_predictions.csv` written by cross_validate.py, so every number
here is recomputed from raw predictions rather than re-quoted from a summary table.

HistGradientBoosting is evaluated on the EXACT SAME patient splits as the neural
models (recovered from each fold's train/val/test prediction files), because a gap of
~0.01 cannot be honestly compared across different fold boundaries.

Usage:
    .venv/Scripts/python.exe scripts/analyze_prereg.py \
        --run-dir results/cv_hard --patients-csv data/gen_hard/patients.csv \
        --scans-csv data/gen_hard/scans.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scripts.train_baselines import build_feature_table

# Decision thresholds -- fixed in PREREGISTRATION.md before the cohort existed.
MIN_GAP = 0.010
SD_MULT = 2.0
MIN_SIGN_AGREE = 4  # of 5 folds

MODEL_DIRS = {
    "FETA-Transformer": "feta",
    "PREG-Net": "preg",
    "Gated hybrid": "hybrid",
    "Logit-average ensemble": "ensemble",
}


def per_fold_auroc(run_dir: Path, sub: str) -> list[float]:
    out = []
    for fold in sorted((run_dir / sub).glob("fold_*")):
        df = pd.read_csv(fold / "test_predictions.csv")
        out.append(roc_auc_score(df["label"], df["probability"]))
    return out


def hgb_same_folds(run_dir: Path, patients: Path, scans: Path) -> list[float]:
    """Train HGB on each fold's own train+val patients, score its test patients."""
    feats = build_feature_table(pd.read_csv(patients), pd.read_csv(scans))
    feats = feats.set_index("patient_id")
    ref = run_dir / "hybrid"
    aucs = []
    for fold in sorted(ref.glob("fold_*")):
        tr = pd.read_csv(fold / "train_predictions.csv")["patient_id"]
        va = pd.read_csv(fold / "val_predictions.csv")["patient_id"]
        te = pd.read_csv(fold / "test_predictions.csv")["patient_id"]
        fit_ids = pd.concat([tr, va]).values
        Xtr = feats.loc[fit_ids].drop(columns=["label"]).values
        ytr = feats.loc[fit_ids, "label"].values
        Xte = feats.loc[te.values].drop(columns=["label"]).values
        yte = feats.loc[te.values, "label"].values
        m = HistGradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--patients-csv", required=True)
    ap.add_argument("--scans-csv", required=True)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)

    scores = {n: per_fold_auroc(run_dir, d) for n, d in MODEL_DIRS.items()}
    scores["HistGradientBoosting"] = hgb_same_folds(
        run_dir, Path(a.patients_csv), Path(a.scans_csv)
    )

    print("=" * 84)
    print(f"PRE-REGISTERED ANALYSIS -- {run_dir}")
    print("=" * 84)
    print(f"{'model':<26}{'AUROC (mean +/- SD)':>22}   per-fold")
    print("-" * 84)
    for n, v in sorted(scores.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v)
        folds = " ".join(f"{x:.3f}" for x in v)
        print(f"{n:<26}{v.mean():>11.4f} +/- {v.std():.4f}   {folds}")

    hyb = np.array(scores["Gated hybrid"])
    baselines = {k: np.array(v) for k, v in scores.items() if k != "Gated hybrid"}
    best_name = max(baselines, key=lambda k: baselines[k].mean())
    best = baselines[best_name]

    d = hyb.mean() - best.mean()
    sd_pooled = float(np.sqrt((hyb.std() ** 2 + best.std() ** 2) / 2))
    per_fold_d = hyb - best
    n_pos = int((per_fold_d > 0).sum())
    sign_agree = max(n_pos, len(per_fold_d) - n_pos)

    print("\n" + "=" * 84)
    print("PRIMARY COMPARISON (decision rule fixed in PREREGISTRATION.md)")
    print("=" * 84)
    print(f"  best same-input baseline : {best_name}  ({best.mean():.4f})")
    print(f"  gated hybrid             : {hyb.mean():.4f}")
    print(f"  gap d                    : {d:+.4f}")
    print(f"  pooled fold SD           : {sd_pooled:.4f}   (2*SD = {SD_MULT*sd_pooled:.4f})")
    print(f"  per-fold gaps            : {' '.join(f'{x:+.3f}' for x in per_fold_d)}")
    print(f"  sign agreement           : {sign_agree}/{len(per_fold_d)} folds "
          f"(need >= {MIN_SIGN_AGREE})")

    supported = (d >= MIN_GAP and d >= SD_MULT * sd_pooled
                 and sign_agree >= MIN_SIGN_AGREE)
    refuted = (d <= -MIN_GAP and abs(d) >= SD_MULT * sd_pooled)

    print("\n  VERDICT:", end=" ")
    if supported:
        print("SUPPORTED -- modest but consistent improvement.")
    elif refuted:
        print("REFUTED -- the gated hybrid UNDERPERFORMS the best same-input baseline.")
    else:
        print("TIE -- gap is within noise. Per the pre-registration this is reported")
        print("           as a TIE, explicitly NOT a win, regardless of its sign.")
    print("=" * 84)


if __name__ == "__main__":
    main()
