"""Is the scan trajectory TABULARIZABLE?

The 13-regime screen said a tree beats every fusion. The suspicion is that this is a
property of the TASK, not the architectures: with 5 regular visits x 4 measures, a
trajectory collapses losslessly into aggregates, so no sequence model can add value.

This measures the ceiling of feature engineering on the scan channel alone, against
the ORACLE sono ceiling (0.820 on gen_hard, from CLAUDE.md). Three feature sets of
increasing richness, all fed to the same HGB:

  agg   -- mean/std/last/min/max + slope   (what compare-style baselines use)
  flat  -- every visit's every measurement as its own column, padded
  rich  -- flat + pairwise ratios + per-interval velocities + GA-conditional residuals

If `rich` approaches 0.820, the trajectory is fully tabularizable and a temporal model
has NOTHING structural to win -- the honest conclusion is the negative one.
If `rich` plateaus well below 0.820, that gap is real sequence structure and a
temporal architecture has genuine headroom. That is a testable positive claim.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")
MEAS = ["FHR", "CRL", "GS", "YSD"]
COH = "data/gen_hard"


ORACLE_SONO_CEILING = 0.820   # measured on gen_hard; see CLAUDE.md


def cv(X, y, tag, verbose=True):
    X = X.astype(float).values if isinstance(X, pd.DataFrame) else X
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    a = roc_auc_score(y, oof)
    if verbose:
        print(f"  {tag:34s} {X.shape[1]:4d} feats   AUROC {a:.3f}", flush=True)
    return a, X.shape[1]


def main():
    pat = pd.read_csv(REPO / COH / "patients.csv")
    sc = pd.read_csv(REPO / COH / "scans.csv")
    ch = pd.read_csv(REPO / COH / "channel_labels.csv")
    y = pat["label"].values
    ids = pat["patient_id"]

    # ---- agg ----
    g = sc.groupby("patient_id")
    agg = g[MEAS + ["gestational_age_weeks"]].agg(["mean", "std", "last", "min", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg["n"] = g.size()
    for m in MEAS:
        d = sc.dropna(subset=[m])
        agg[f"{m}_slope"] = d.groupby("patient_id").apply(
            lambda t: np.polyfit(t["gestational_age_weeks"], t[m], 1)[0]
            if len(t) > 1 and t["gestational_age_weeks"].nunique() > 1 else np.nan,
            include_groups=False)
    agg = agg.reindex(ids).reset_index(drop=True)

    # ---- flat ----
    sc = sc.sort_values(["patient_id", "gestational_age_weeks"])
    sc["visit"] = sc.groupby("patient_id").cumcount()
    flat = sc.pivot_table(index="patient_id", columns="visit",
                          values=MEAS + ["gestational_age_weeks"])
    flat.columns = [f"{a}_v{b}" for a, b in flat.columns]
    flat = flat.reindex(ids).reset_index(drop=True)

    # ---- rich: velocities, ratios, GA-conditional residuals ----
    rich = flat.copy()
    for m in MEAS:
        cols = [c for c in flat.columns if c.startswith(f"{m}_v")]
        gcols = [f"gestational_age_weeks_v{i}" for i in range(len(cols))]
        for i in range(1, len(cols)):
            dt = flat[gcols[i]] - flat[gcols[i - 1]]
            rich[f"{m}_vel{i}"] = (flat[cols[i]] - flat[cols[i - 1]]) / dt.replace(0, np.nan)
    for a in MEAS:
        for b in MEAS:
            if a < b:
                rich[f"{a}_{b}_ratio"] = flat[f"{a}_v0"] / flat[f"{b}_v0"].replace(0, np.nan)
    # residual of each measure vs a pooled GA regression = "smaller than expected for dates"
    for m in MEAS:
        long = sc.dropna(subset=[m, "gestational_age_weeks"])
        co = np.polyfit(long["gestational_age_weeks"], long[m], 2)
        r = long.assign(res=long[m] - np.polyval(co, long["gestational_age_weeks"]))
        gr = r.groupby("patient_id")["res"]
        for stat in ["mean", "min", "last"]:
            rich[f"{m}_res_{stat}"] = gr.agg(stat).reindex(ids).values

    print(f"\ncohort {COH}  n={len(y)}  prevalence={y.mean():.3f}")
    print("\n--- predicting the OBSERVED label from scans only ---")
    a_agg, n_agg = cv(agg, y, "agg (mean/std/last/slope)")
    a_flat, n_flat = cv(flat, y, "flat (every visit x measure)")
    a_rich, n_rich = cv(rich, y, "rich (+velocity/ratio/GA-residual)")

    # oracle: how well can scans recover their OWN channel? diagnostics only.
    ysono = ch.set_index("patient_id").reindex(ids)["label_sono"].values.astype(int)
    print("\n--- recovering label_sono from scans (diagnostics, never a model input) ---")
    b_rich, _ = cv(rich, ysono, "rich -> label_sono")

    best = max(a_agg, a_flat, a_rich)
    print("\n" + "=" * 66)
    print(f"  feature-engineering ceiling on scans : {best:.3f}")
    print(f"  ORACLE sono ceiling (CLAUDE.md)      : {ORACLE_SONO_CEILING:.3f}")
    print(f"  headroom a temporal model could win  : {ORACLE_SONO_CEILING - best:+.3f}")
    print(f"  scans -> own channel (label_sono)    : {b_rich:.3f}")
    print("=" * 66)

    return dict(
        sets=[("Aggregates\n(mean/std/last/slope)", n_agg, a_agg),
              ("Per-visit\nflattening", n_flat, a_flat),
              ("+ velocities, ratios,\nGA-residuals", n_rich, a_rich)],
        oracle=ORACLE_SONO_CEILING, sono_recovery=b_rich,
        cohort=COH, n=len(y), prevalence=float(y.mean()))


if __name__ == "__main__":
    main()
