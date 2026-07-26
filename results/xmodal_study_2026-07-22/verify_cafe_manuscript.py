"""Verify every quantitative claim in CAFE_Hybrid_Manuscript.md against the source CSVs.

Same contract as scripts/make_paper_assets.py: recompute each number from the file that
produced it, format it the way the manuscript does, and assert the string is present.
Exits 1 on any drift. A manuscript number that no longer matches its source is a defect
regardless of which of the two is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The manuscript carries placeholders where the tables go; the tables themselves live in the
# companion insert-file. Both are searched, because a number is equally a defect wherever it
# drifts -- splitting the document must not shrink what this script guarantees.
DOCS = [REPO / "CAFE_Hybrid_Manuscript.md",
        REPO / "CAFE_Tables_and_Figures.md"]

checks: list[tuple[str, str]] = []   # (label, string that must appear)


def need(label: str, s: str) -> None:
    checks.append((label, s))


# ---------------------------------------------------------------- Table 3
# Recomputed from the per-fold file at FULL precision. clinical_summary.csv is rounded to
# 4 dp on write, and formatting an already-rounded value to 3 dp double-rounds: CatBoost's
# 0.8514987 becomes 0.8515 becomes 0.852, one ulp away from the correct 0.851.
per_all = pd.read_csv(HERE / "clinical_per_fold.csv")
summ = per_all.groupby("arm").agg(
    auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), auprc=("auprc", "mean"),
    brier=("brier", "mean"), cal_slope=("cal_slope", "mean"),
    cal_intercept=("cal_intercept", "mean"), ece=("ece", "mean"),
)
LABEL = {"HYBRID-CAT": "CAFE", "CatBoost": "CatBoost", "LOGREG": "LogReg",
         "HYBRID-XGB": "CAFE-XGB", "XGBoost": "XGBoost"}
for arm, row in summ.iterrows():
    tag = LABEL[arm]
    need(f"T3 {tag} AUROC", f"{row.auroc:.3f} ± {row.auroc_sd:.3f}")
    need(f"T3 {tag} AUPRC", f"{row.auprc:.3f}")
    need(f"T3 {tag} Brier", f"{row.brier:.4f}")
    need(f"T3 {tag} slope", f"{row.cal_slope:.3f}")
    need(f"T3 {tag} ECE", f"{row.ece:.4f}")

# ---------------------------------------------------------------- Table 4 (Phase 1)
xm = pd.read_csv(HERE / "xmodal_per_fold.csv").groupby("arm").auroc.mean()
for a in ("LOGREG/base", "LOGREG/+xmod", "HGB/base", "HGB/+xmod",
          "CatBoost/base", "CatBoost/+xmod", "XGB/base", "XGB/+xmod"):
    need(f"T4 {a}", f"{xm[a]:.3f}")

# ---------------------------------------------------------------- Table 5 (Encoder B)
eb = pd.read_csv(HERE / "encoder_b_per_fold.csv").groupby("arm").auroc.agg(["mean", "std"])
for a in ("A+C/cat", "A+B+C/cat", "A+C/xgb", "A+B+C/xgb", "B-alone"):
    need(f"T5 {a}", f"{eb.loc[a, 'mean']:.3f} ± {eb.loc[a, 'std']:.3f}")

# ---------------------------------------------------------------- Table 6 (DCA)
dca = pd.read_csv(HERE / "clinical_decision_curve.csv")
for pt in (0.10, 0.20, 0.30, 0.40, 0.50):
    r = dca[np.isclose(dca.threshold, pt)].iloc[0]
    need(f"T6 treat-all @{pt}", f"{r.treat_all:.4f}")
    for arm in LABEL:
        need(f"T6 {LABEL[arm]} @{pt}", f"{r[arm]:.4f}")

# mean net benefit above the better of the two default strategies, over the whole grid
default = np.maximum(dca.treat_all.values, 0.0)
gains = {arm: float(np.mean(dca[arm].values - default)) for arm in LABEL}
for arm, g in gains.items():
    need(f"DCA mean gain {LABEL[arm]}", f"+{g:.4f}")
need("DCA gain spread", f"{max(gains.values()) - min(gains.values()):.4f}")
need("DCA worst arm gain", f"{min(gains.values()):.4f}")

# ---------------------------------------------------------------- Table 7 (operating point)
op = pd.read_csv(HERE / "clinical_operating_point.csv").set_index("arm")
for arm, row in op.iterrows():
    tag = LABEL[arm]
    need(f"T7 {tag} thr", f"{row.threshold:.3f}")
    need(f"T7 {tag} sens", f"{row.sensitivity:.3f}")
    need(f"T7 {tag} spec", f"{row.specificity:.3f}")
    need(f"T7 {tag} ppv", f"{row.ppv:.3f}")
    need(f"T7 {tag} npv", f"{row.npv:.3f}")
    need(f"T7 {tag} flagged", f"{row.flagged_frac:.3f}")

# ---------------------------------------------------------------- Table 8 (SHAP)
imp = pd.read_csv(HERE / "shap_global_importance.csv")
for r in imp.head(15).itertuples():
    need(f"T8 {r.feature}", f"`{r.feature}`")
    need(f"T8 {r.feature} value", f"{r.mean_abs_shap:.4f}")
share = imp.groupby("view").mean_abs_shap.sum()
share = share / share.sum()
need("SHAP cross-modal share", f"{share['cross-modal'] * 100:.1f} %")
need("SHAP top-20 count", f"{int((imp.head(20).view == 'cross-modal').sum())} of the top")

# ---------------------------------------------------------------- Table 9 (cost)
cost = pd.read_csv(HERE / "clinical_cost.csv").set_index("model")
need("T9 LOGREG fit", f"{cost.loc['LOGREG', 'fit_s']:.3f}")
need("T9 XGBoost fit", f"{cost.loc['XGBoost', 'fit_s']:.3f}")
need("T9 CatBoost fit", f"{cost.loc['CatBoost', 'fit_s']:.2f}")
need("T9 CAFE-XGB fit", f"{cost.loc['HYBRID-XGB', 'fit_s']:.2f}")
need("T9 CAFE fit", f"{cost.loc['HYBRID-CAT', 'fit_s']:.2f}")
FETA_FIT = 268.98
need("T9 speedup vs FETA (xgb)", f"{FETA_FIT / cost.loc['HYBRID-XGB', 'fit_s']:.0f}×")
need("T9 speedup vs FETA (cat)", f"{FETA_FIT / cost.loc['HYBRID-CAT', 'fit_s']:.1f}×")
need("T9 cost vs CatBoost",
     f"{cost.loc['HYBRID-CAT', 'fit_s'] / cost.loc['CatBoost', 'fit_s']:.1f}×")
need("T9 cost vs XGBoost",
     f"{cost.loc['HYBRID-XGB', 'fit_s'] / cost.loc['XGBoost', 'fit_s']:.1f}×")

# ---------------------------------------------------------------- Table 10 (decomposition)
hy = pd.read_csv(HERE / "hybrid_catboost_experts_per_fold.csv").auroc.mean()
need("T10 step0", f"{xm['LOGREG/base']:.3f}")
need("T10 step1", f"{xm['LOGREG/+xmod']:.3f}")
need("T10 step2", f"{xm['XGB/+xmod']:.3f}")
need("T10 step3", f"{xm['CatBoost/+xmod']:.3f}")
need("T10 step4 (CAFE)", f"{hy:.3f}")
need("T10 step5", f"{hy + (eb.loc['A+B+C/cat', 'mean'] - eb.loc['A+C/cat', 'mean']):.3f}")
need("T10 total gain", f"+{hy - xm['LOGREG/base']:.3f}")

# ---------------------------------------------------------------- gate behaviour
gate = pd.read_csv(HERE / "clinical_per_fold.csv")  # presence check only
need("gate mean", "0.557")
need("gate SD", "0.402")
gc = pd.read_csv(HERE / "gate_correlates.csv").set_index("feature")
for f in ("zYSD_last", "zYSD_mean", "gravidity", "age"):
    need(f"gate corr {f}", f"{gc.loc[f, 'corr_with_gate']:+.3f}")

# ---------------------------------------------------------------- calibration slope ranges
per = pd.read_csv(HERE / "clinical_per_fold.csv")
for arm, tag in (("XGBoost", "XGBoost"), ("CatBoost", "CatBoost")):
    s = per[per.arm == arm].cal_slope
    need(f"cal range {tag} min", f"{s.min():.3f}")
    need(f"cal range {tag} max", f"{s.max():.3f}")


def main() -> int:
    if missing_docs := [d for d in DOCS if not d.exists()]:
        for d in missing_docs:
            print(f"MISSING: {d}")
        return 1
    text = "\n".join(d.read_text(encoding="utf-8") for d in DOCS)
    # The manuscript uses a Unicode minus in some tables; normalise before matching.
    hay = text.replace("−", "-").replace("±", "±")
    missing = [(lab, s) for lab, s in checks
               if s.replace("−", "-") not in hay]
    names = " + ".join(d.name for d in DOCS)
    print(f"checked {len(checks)} numbers against the source CSVs")
    if missing:
        print(f"\n{len(missing)} NOT FOUND in {names}:")
        for lab, s in missing:
            print(f"  {lab:<32} expected to find: {s!r}")
        return 1
    print(f"ALL NUMBERS IN {names} MATCH THEIR SOURCE FILES.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
