"""Can the guide's hybrid honestly reach the predicted 0.840-0.855 band?

The first run built the hybrid's two experts on XGBoost, which is the WEAKEST tree on
this cohort (0.830). CatBoost is 0.850. So the hybrid was handicapped by its component,
not by its architecture -- and rebuilding it on CatBoost is a legitimate improvement,
not sandbagging, because CatBoost is already an arm in the baseline set. Rule 7 forbids
weakening baselines to make the hybrid win; it does not forbid making the hybrid better,
provided the baseline keeps the same component and the same budget. It does here: the
comparator is CatBoost on the union of both views.

Also reports whether the gate is actually PER-PATIENT. A softmax gate that collapses to
a near-constant weight is a fixed blend wearing a gate's clothing, and the paper should
say so. This is the measurement behind the guide's Phase-4 "which encoder was trusted".

Comparators are read from xmodal_per_fold.csv rather than re-run: identical folds,
identical seeds, fully deterministic, so re-running would only risk divergence.

DECISION RULE -- unchanged from xmodal_features.py, applied verbatim:
    REAL only if |mean delta| > 2 x pooled fold SD AND sign agrees on >= 8/10 folds.
    Otherwise TIE, whatever the sign.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "regime_study_2026-07-19"))

from catboost import CatBoostClassifier  # noqa: E402
from published_families_benchmark import COH, load  # noqa: E402
from xmodal_features import SEED, build_xmodal_block, guide_hybrid, verdict  # noqa: E402

TARGET_LO, TARGET_HI = 0.840, 0.855


def cat() -> CatBoostClassifier:
    return CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False)


def main() -> None:
    pat = pd.read_csv(REPO / COH / "patients.csv")
    raw = pd.read_csv(REPO / COH / "scans.csv")
    _, Xs, y, Xbase = load(COH)
    Xmod = np.nan_to_num(build_xmodal_block(pat, raw).values,
                         nan=0.0, posinf=0.0, neginf=0.0)

    splits = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(Xs, y)
    )
    print(f"cohort {COH}   n={len(y)}   5x2 CV, folds identical to the published run")
    print("hybrid rebuilt on CatBoost experts (was XGBoost)\n")

    auroc, auprc, gate_sd, gate_mean = [], [], [], []
    for k, (tr, te) in enumerate(splits, 1):
        p, w0 = guide_hybrid(Xbase, Xmod, y, tr, te, seed=0, expert=cat,
                             return_gate=True)
        auroc.append(roc_auc_score(y[te], p))
        auprc.append(average_precision_score(y[te], p))
        gate_mean.append(float(w0.mean()))
        gate_sd.append(float(w0.std()))
        print(f"  fold {k:2d}/10  auroc={auroc[-1]:.3f}  "
              f"gate w0={gate_mean[-1]:.3f}+/-{gate_sd[-1]:.3f}", flush=True)

    prev = pd.read_csv(HERE / "xmodal_per_fold.csv").pivot(
        index="fold", columns="arm", values="auroc")
    m, sd = float(np.mean(auroc)), float(np.std(auroc))
    se = float(np.std(auroc, ddof=1) / np.sqrt(len(auroc)))

    print("\n" + "=" * 68)
    print(f"HYBRID / CatBoost experts   {m:.3f} +/- {sd:.3f}   "
          f"AUPRC {np.mean(auprc):.3f}")
    print(f"  95% CI on the mean        [{m - 1.96*se:.3f}, {m + 1.96*se:.3f}]")
    print(f"  guide's predicted band    [{TARGET_LO:.3f}, {TARGET_HI:.3f}]")
    inband = TARGET_LO <= m <= TARGET_HI
    print(f"  point estimate in band?   {'YES' if inband else 'NO'}")
    print(f"  folds >= {TARGET_LO}:          {int((np.array(auroc) >= TARGET_LO).sum())}/10")
    print("=" * 68)

    print("\nvs the same-component baseline and the previous hybrid")
    print(f"{'comparison':<38}{'delta':>8}{'2*SD':>8}{'sign':>7}  verdict")
    print("-" * 68)
    rows = []
    for name in ("CatBoost/+xmod", "CatBoost/base", "GUIDE-HYBRID"):
        v = verdict(auroc, prev[name].values)
        rows.append(dict(comparison=f"HYBRID-CAT - {name}", **v))
        print(f"{'HYBRID-CAT - ' + name:<38}{v['delta']:>+8.3f}"
              f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    print("\nIS THE GATE ACTUALLY PER-PATIENT?")
    print(f"  mean weight on the base expert   {np.mean(gate_mean):.3f}")
    print(f"  within-fold SD across patients   {np.mean(gate_sd):.4f}")
    print("  a per-patient gate needs spread here; ~0 means a fixed blend.")

    pd.DataFrame(dict(fold=range(1, 11), auroc=auroc, auprc=auprc,
                      gate_w0_mean=gate_mean, gate_w0_sd=gate_sd)).to_csv(
        HERE / "hybrid_catboost_experts_per_fold.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "hybrid_catboost_experts_verdicts.csv", index=False)
    print("\nwrote hybrid_catboost_experts_per_fold.csv and _verdicts.csv")


if __name__ == "__main__":
    main()
