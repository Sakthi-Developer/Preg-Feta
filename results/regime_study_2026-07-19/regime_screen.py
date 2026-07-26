"""Fast regime screen: where can a PER-PATIENT GATE beat a FIXED-WEIGHT fusion?

Real FETA/PREG-Net training is far too slow to sweep regimes (>1400 min/fold in the
missingness log). So each tower is stood in by a cheap surrogate with the SAME
information diet:

    sono tower  -> HGB on scan-derived features only
    mat  tower  -> HGB on maternal features only

Then four fusions are compared, all out-of-fold, all on identical folds:

    AVG    fixed logit average                 (what the gate must beat)
    STACK  fusion on [l_sono, l_mat] only      (learned but context-BLIND weights)
    GATE   fusion on [l_sono, l_mat, context]  (context-AWARE = the gate's mechanism)
    JOINT  HGB on all raw features             (the off-the-shelf baseline that has
                                                tied everything so far)

GATE - STACK isolates the value of *per-patient* routing specifically, with tower
quality held constant. That difference is the paper's claim in miniature. A regime is
worth a real CV run only if GATE beats both AVG and JOINT.

Surrogates are not the real models -- this is a FEASIBILITY screen to decide where to
spend a real run, not a result to report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")
SEED = 0
NFOLD = 5


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build(cohort: str):
    """-> maternal frame, scan frame, context frame, labels. NaNs preserved (HGB eats them)."""
    pat = pd.read_csv(REPO / cohort / "patients.csv")
    sc = pd.read_csv(REPO / cohort / "scans.csv")

    y = pat["label"].values
    mat = pat.drop(columns=["patient_id", "label"])
    cat = [c for c in mat.columns if not pd.api.types.is_numeric_dtype(mat[c])]
    mat = pd.get_dummies(mat, columns=cat, dummy_na=True) if cat else mat
    mat = mat.astype(float)

    meas = ["FHR", "CRL", "GS", "YSD"]
    g = sc.groupby("patient_id")
    scan = g[meas + ["gestational_age_weeks"]].agg(["mean", "std", "last", "min", "max"])
    scan.columns = ["_".join(c) for c in scan.columns]
    scan["n_visits"] = g.size()
    # growth slope per measure -- the trajectory signal FETA is meant to read
    for m in meas:
        d = sc.dropna(subset=[m])
        slope = d.groupby("patient_id").apply(
            lambda t: np.polyfit(t["gestational_age_weeks"], t[m], 1)[0]
            if len(t) > 1 and t["gestational_age_weeks"].nunique() > 1 else np.nan,
            include_groups=False)
        scan[f"{m}_slope"] = slope
    scan = scan.reindex(pat["patient_id"]).reset_index(drop=True).astype(float)

    # CONTEXT the gate is allowed to route on: how much of each modality is present.
    # Deliberately contains NO outcome information -- only data-availability structure.
    ctx = pd.DataFrame({
        "n_visits": scan["n_visits"].fillna(0),
        "scan_missing_frac": scan[[f"{m}_mean" for m in meas]].isna().mean(axis=1),
        "mat_missing_frac": mat.isna().mean(axis=1),
        "ga_span": (scan["gestational_age_weeks_max"] - scan["gestational_age_weeks_min"]),
    })
    return mat, scan, ctx, y


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED)


def screen(name, mat, scan, ctx, y, folds=NFOLD):
    n = len(y)
    oof = {k: np.zeros(n) for k in
           ["sono", "matr", "stack", "stack_h", "gate", "gate_lr", "joint"]}
    allX = pd.concat([mat, scan], axis=1)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(np.zeros(n), y):
        ls = np.zeros(n)
        lm = np.zeros(n)
        # towers
        m1 = hgb().fit(scan.iloc[tr], y[tr]); oof["sono"][te] = m1.predict_proba(scan.iloc[te])[:, 1]
        m2 = hgb().fit(mat.iloc[tr], y[tr]);  oof["matr"][te] = m2.predict_proba(mat.iloc[te])[:, 1]
        oof["joint"][te] = hgb().fit(allX.iloc[tr], y[tr]).predict_proba(allX.iloc[te])[:, 1]

        # inner OOF tower logits on train, so the fusers are not fit on leaked scores
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
        tr_s, tr_m = np.zeros(len(tr)), np.zeros(len(tr))
        ytr = y[tr]
        for a, b in inner.split(np.zeros(len(tr)), ytr):
            ia, ib = tr[a], tr[b]
            tr_s[b] = hgb().fit(scan.iloc[ia], y[ia]).predict_proba(scan.iloc[ib])[:, 1]
            tr_m[b] = hgb().fit(mat.iloc[ia], y[ia]).predict_proba(mat.iloc[ib])[:, 1]

        Ztr = np.column_stack([logit(tr_s), logit(tr_m)])
        Zte = np.column_stack([logit(oof["sono"][te]), logit(oof["matr"][te])])
        Gtr = np.column_stack([Ztr, ctx.iloc[tr].fillna(-1).values])
        Gte = np.column_stack([Zte, ctx.iloc[te].fillna(-1).values])

        # 2x2 design: {LogReg, HGB} x {logits only, logits + context}. Comparing
        # across the learner axis would confound "routing helps" with "HGB overfits
        # a 2-feature problem", so the context effect is read DOWN each column with
        # the learner held fixed.
        lr = lambda: LogisticRegression(max_iter=1000)
        oof["stack"][te] = lr().fit(Ztr, ytr).predict_proba(Zte)[:, 1]        # LR  , no ctx
        oof["stack_h"][te] = hgb().fit(Ztr, ytr).predict_proba(Zte)[:, 1]     # HGB , no ctx
        oof["gate"][te] = hgb().fit(Gtr, ytr).predict_proba(Gte)[:, 1]        # HGB , ctx
        oof["gate_lr"][te] = lr().fit(Gtr, ytr).predict_proba(Gte)[:, 1]      # LR  , ctx

    avg = (logit(oof["sono"]) + logit(oof["matr"])) / 2
    res = {
        "regime": name, "n": n, "prev": round(float(y.mean()), 3),
        "sono": roc_auc_score(y, oof["sono"]), "mat": roc_auc_score(y, oof["matr"]),
        "AVG": roc_auc_score(y, avg),
        "sLR": roc_auc_score(y, oof["stack"]), "gLR": roc_auc_score(y, oof["gate_lr"]),
        "sHGB": roc_auc_score(y, oof["stack_h"]), "gHGB": roc_auc_score(y, oof["gate"]),
        "JOINT": roc_auc_score(y, oof["joint"]),
    }
    # context effect with the LEARNER HELD FIXED -- the clean read of "does per-patient
    # routing help?". Positive = routing buys something.
    res["ctx@LR"] = res["gLR"] - res["sLR"]
    res["ctx@HGB"] = res["gHGB"] - res["sHGB"]
    res["best_fuse-AVG"] = max(res["gLR"], res["gHGB"]) - res["AVG"]
    res["best_fuse-JOINT"] = max(res["gLR"], res["gHGB"]) - res["JOINT"]
    return res


# ---------------- regime constructors ----------------

def sub(mat, scan, ctx, y, idx):
    return (mat.iloc[idx].reset_index(drop=True), scan.iloc[idx].reset_index(drop=True),
            ctx.iloc[idx].reset_index(drop=True), y[idx])


def main():
    out = []
    base = build("data/gen_hard")

    # A. missingness ladder (cohorts already on disk)
    for tag, coh in [("clean", "data/gen_hard"),
                     ("miss-light", "data/gen_hard_miss_light"),
                     ("miss-moderate", "data/gen_hard_miss_moderate"),
                     ("miss-heavy", "data/gen_hard_miss_heavy")]:
        if (REPO / coh).exists():
            out.append(screen(f"A/{tag}", *build(coh)))
            print(pd.DataFrame(out[-1:]).round(3).to_string(index=False), flush=True)

    mat, scan, ctx, y = base
    rng = np.random.default_rng(SEED)

    # B. per-patient scan degradation: HALF the cohort has its scan channel corrupted.
    #    A gate should route those patients to the maternal tower; a fixed average cannot.
    for frac in (0.3, 0.5):
        m2, s2, c2 = mat.copy(), scan.copy(), ctx.copy()
        bad = rng.random(len(y)) < frac
        cols = [c for c in s2.columns if c != "n_visits"]
        s2.loc[bad, cols] = s2.loc[bad, cols] + rng.normal(
            0, 3 * s2[cols].std().values, size=(bad.sum(), len(cols)))
        c2["scan_corrupt_hint"] = bad.astype(float)  # clinics DO know which scans are poor
        out.append(screen(f"B/scan-noise-{frac:g}", m2, s2, c2, y))
        print(pd.DataFrame(out[-1:]).round(3).to_string(index=False), flush=True)

    # C. per-patient MODALITY DROPOUT: a fraction lose scans entirely, another lose
    #    most maternal fields. Which tower is usable now varies BY PATIENT.
    for frac in (0.3, 0.5):
        m2, s2, c2 = mat.copy(), scan.copy(), ctx.copy()
        r = rng.random(len(y))
        no_scan, no_mat = r < frac / 2, (r >= frac / 2) & (r < frac)
        s2.loc[no_scan, :] = np.nan
        m2.loc[no_mat, :] = np.nan
        c2 = c2.copy()
        c2["scan_missing_frac"] = s2.isna().mean(axis=1)
        c2["mat_missing_frac"] = m2.isna().mean(axis=1)
        out.append(screen(f"C/modality-dropout-{frac:g}", m2, s2, c2, y))
        print(pd.DataFrame(out[-1:]).round(3).to_string(index=False), flush=True)

    # D. small-N (does fusion structure pay off when data is scarce?)
    for n in (400, 800, 1500):
        idx = rng.permutation(len(y))[:n]
        out.append(screen(f"D/small-n-{n}", *sub(mat, scan, ctx, y, idx)))
        print(pd.DataFrame(out[-1:]).round(3).to_string(index=False), flush=True)

    # E. low prevalence (rare-event regime)
    for prev in (0.10, 0.05):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = int(len(neg) * prev / (1 - prev))
        idx = np.concatenate([rng.permutation(pos)[:keep], neg])
        out.append(screen(f"E/prevalence-{prev:g}", *sub(mat, scan, ctx, y, idx)))
        print(pd.DataFrame(out[-1:]).round(3).to_string(index=False), flush=True)

    df = pd.DataFrame(out)
    df.to_csv(Path(__file__).parent / "regime_screen.csv", index=False)
    print("\n" + "=" * 100)
    print(df.round(3).to_string(index=False))
    print("=" * 100)
    print("\nContext effect with learner held fixed (positive = routing helps):")
    print(df.sort_values("ctx@LR", ascending=False)[
        ["regime", "AVG", "sLR", "gLR", "sHGB", "gHGB", "JOINT",
         "ctx@LR", "ctx@HGB", "best_fuse-AVG", "best_fuse-JOINT"]
    ].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
