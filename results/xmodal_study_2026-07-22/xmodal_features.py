"""CROSS-MODAL FEATURE ENGINEERING + GATED FEATURE-VIEW EXPERTS

The one genuinely untested part of the "pivot to a hybrid" proposal.

WHAT IS OPEN, AND WHAT IS NOT
-----------------------------
`tabularizable.py` already tested the SCAN-ONLY feature families -- per-interval
velocities, pairwise ratios, GA-conditional residuals. 30 feats 0.732 -> 59 feats
0.728 against a measured oracle sono ceiling of 0.820. Flat. Those rows are closed.

Never built, therefore genuinely open, are the CROSS-MODAL families:

  Phase 1   discordance signals   (FHR-CRL correlation, trend divergence, z-spread)
            maternal x sonographic interactions  (age x FHR, BMI x CRL, prior x FHR)
            a maternal composite risk score
            -- the scan-only probe could not have measured any of these.

  Phase 2   split those features across two experts and combine them with a learned
            per-patient gate, vs one model on the union.
            CLAUDE.md's dead-end table has "two-tower split at all -> joint tree wins
            13/13", but that was towers over MODALITIES. This is experts over FEATURE
            VIEWS. Close enough to expect the same answer, cheap enough to just run.

DESIGN
------
Rule 2  Base table is build_feature_table_fast, reached through the benchmark's own
        load(). Ad-hoc features understate the tree by ~0.019 and have manufactured a
        false win twice; that path is deliberately not reachable from here.
Rule 5  5x2 RepeatedStratifiedKFold(random_state=0) -- the SAME folds as
        published_families_benchmark, so every number here is directly comparable to
        the published table. HGB/base reproducing 0.832 is the harness self-check.
Rule 7  Identical feature budget for every arm. The cross-modal block is offered to
        LOGREG, HGB, XGBoost and CatBoost alike -- not only to the hybrid.
Rule 3  channel_labels.csv is never opened by this file.
Leakage No label of any kind enters feature construction. Every transform here is
        label-free (GA-conditional z-scores condition on gestational age, not outcome),
        so computing them once outside the fold loop is transductive-but-clean, which
        is the same standing as the existing `rich` set.

DECISION RULE -- fixed before the run, applied unchanged afterwards
------------------------------------------------------------------
The feature effect is read DOWN a column with the LEARNER HELD FIXED. Never across
learners: that exact confound faked a -0.067 "routing penalty" once already.

    A difference is REAL only if
        |mean delta over the 10 folds| > 2 x pooled fold SD
        AND sign agrees with the mean on >= 8/10 folds.
    Everything else is a TIE, reported as a tie, whatever the sign of the point
    estimate and whatever would be more convenient.

Phase 1 verdict = delta(+xmodal - base), per learner.
Phase 2 verdict = delta(GUIDE-HYBRID - XGB/+xmodal): identical information, split and
                  gated versus joint.

AUROC and AUPRC only. Bare accuracy is meaningless here -- all-negative scores 79.7%.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results" / "regime_study_2026-07-19"))

from published_families_benchmark import COH, load  # noqa: E402

from catboost import CatBoostClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

MEAS = ["FHR", "CRL", "GS", "YSD"]
SEED = 42


# ---------------------------------------------------------------- features
def _zs(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0)


def build_xmodal_block(pat: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    """Cross-modal features. Label-free by construction -- `label` is never read."""
    sc = scans.sort_values(["patient_id", "gestational_age_weeks"]).copy()
    pid = sc["patient_id"]

    # GA-conditional z-scores: "is this measurement normal FOR THIS GESTATIONAL WEEK".
    sc["ga_bin"] = sc["gestational_age_weeks"].round().astype(int)
    for m in MEAS:
        grp = sc.groupby("ga_bin")[m]
        mu, sd = grp.transform("mean"), grp.transform("std")
        sc[f"z{m}"] = ((sc[m] - mu) / sd.replace(0, np.nan)).fillna(0.0)

    g = sc.groupby("patient_id", sort=False)
    out = pd.DataFrame(index=g.size().index)

    for m in MEAS:
        out[f"z{m}_last"] = g[f"z{m}"].last()
        out[f"z{m}_mean"] = g[f"z{m}"].mean()
        out[f"z{m}_min"] = g[f"z{m}"].min()

    # OLS slope of each z-trajectory over GA (vectorised: cov/var).
    x = sc["gestational_age_weeks"]
    xm = x.groupby(pid, sort=False).transform("mean")
    den = ((x - xm) ** 2).groupby(pid, sort=False).sum()
    for m in MEAS:
        yv = sc[f"z{m}"]
        ym = yv.groupby(pid, sort=False).transform("mean")
        num = ((x - xm) * (yv - ym)).groupby(pid, sort=False).sum()
        out[f"z{m}_slope"] = np.where(den > 1e-12, num / den.replace(0, np.nan), 0.0)

    # ---- discordance: do the sonographic channels disagree with each other? ----
    def within_patient_corr(a: str, b: str) -> np.ndarray:
        ya, yb = sc[f"z{a}"], sc[f"z{b}"]
        yam = ya.groupby(pid, sort=False).transform("mean")
        ybm = yb.groupby(pid, sort=False).transform("mean")
        num = ((ya - yam) * (yb - ybm)).groupby(pid, sort=False).sum()
        da = ((ya - yam) ** 2).groupby(pid, sort=False).sum()
        db = ((yb - ybm) ** 2).groupby(pid, sort=False).sum()
        d = np.sqrt(da * db)
        return np.where(d > 1e-12, num / d.replace(0, np.nan), 0.0)

    for a, b in [("FHR", "CRL"), ("CRL", "GS"), ("FHR", "GS"), ("CRL", "YSD")]:
        out[f"corr_{a}_{b}"] = within_patient_corr(a, b)
        out[f"disc_last_{a}_{b}"] = out[f"z{a}_last"] - out[f"z{b}_last"]
        out[f"disc_slope_{a}_{b}"] = out[f"z{a}_slope"] - out[f"z{b}_slope"]

    out["disc_sign_FHR_CRL"] = (
        np.sign(out["zFHR_slope"]) != np.sign(out["zCRL_slope"])
    ).astype(float)
    zlast = out[[f"z{m}_last" for m in MEAS]]
    out["disc_spread"] = zlast.max(axis=1) - zlast.min(axis=1)
    out["disc_worst_channel"] = zlast.min(axis=1)

    # ---- maternal composite + maternal x sonographic interactions ----
    p = pat.set_index("patient_id").reindex(out.index)
    ivf = (p["conception"].astype(str).str.upper() == "IVF").astype(float)
    nonsingle = 1.0 - p["singleton"].astype(float)
    out["mat_composite"] = (
        _zs(p["age"].astype(float))
        + _zs(p["bmi"].astype(float))
        + _zs(p["previous_loss"].astype(float))
        + ivf
        + nonsingle
    ).values

    age_z = _zs(p["age"].astype(float)).values
    bmi_z = _zs(p["bmi"].astype(float)).values
    grav_z = _zs(p["gravidity"].astype(float)).values
    par_z = _zs(p["parity"].astype(float)).values
    prev = p["previous_loss"].astype(float).values

    out["age_x_zFHR_last"] = age_z * out["zFHR_last"]
    out["bmi_x_zCRL_last"] = bmi_z * out["zCRL_last"]
    out["prevloss_x_zFHR_last"] = prev * out["zFHR_last"]
    out["age_x_zCRL_slope"] = age_z * out["zCRL_slope"]
    out["bmi_x_zGS_last"] = bmi_z * out["zGS_last"]
    out["grav_x_zFHR_mean"] = grav_z * out["zFHR_mean"]
    out["parity_x_zCRL_last"] = par_z * out["zCRL_last"]
    out["ivf_x_zCRL_slope"] = ivf.values * out["zCRL_slope"]
    out["comp_x_zFHR_last"] = out["mat_composite"] * out["zFHR_last"]
    out["comp_x_zCRL_slope"] = out["mat_composite"] * out["zCRL_slope"]
    out["comp_x_disc_spread"] = out["mat_composite"] * out["disc_spread"]

    return out.reindex(pat["patient_id"]).astype(float)


# ---------------------------------------------------------------- learners
def xgb() -> XGBClassifier:
    return XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0)


def learners() -> dict:
    return {
        "LOGREG": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "HGB": lambda: HistGradientBoostingClassifier(random_state=SEED),
        "XGB": xgb,
        "CatBoost": lambda: CatBoostClassifier(
            random_state=SEED, verbose=0, allow_writing_files=False
        ),
    }


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def guide_hybrid(Xb, Xx, y, tr, te, seed: int = 0, expert=None, return_gate=False):
    """The guide's Phase-2 architecture, implemented in its strongest honest form.

    Expert A on the base view, Expert C on the cross-modal view, combined by a learned
    per-patient softmax gate over the two experts' logits.

    `expert` is a zero-arg factory for the base learner, so the architecture can be
    rebuilt on a stronger component without touching the fusion logic. It defaults to
    XGBoost; passing CatBoost is a legitimate improvement rather than sandbagging,
    because CatBoost is already an arm in the baseline set.

    The gate trains on INNER out-of-fold expert logits. Training it on in-fold logits
    would teach it to trust whichever expert overfits harder, which flatters the
    architecture -- this is the sandbagging-in-reverse version of rule 7.
    """
    expert = expert or xgb
    Xb_tr, Xx_tr, y_tr = Xb[tr], Xx[tr], y[tr]
    oof_b, oof_x = np.zeros(len(tr)), np.zeros(len(tr))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for itr, ite in inner.split(Xb_tr, y_tr):
        oof_b[ite] = _logit(
            expert().fit(Xb_tr[itr], y_tr[itr]).predict_proba(Xb_tr[ite])[:, 1]
        )
        oof_x[ite] = _logit(
            expert().fit(Xx_tr[itr], y_tr[itr]).predict_proba(Xx_tr[ite])[:, 1]
        )

    # Gate sees the patient's own features, exactly as a per-patient gate is meant to.
    sc = StandardScaler().fit(Xb_tr)
    Gtr = torch.tensor(sc.transform(Xb_tr), dtype=torch.float32)
    Lb = torch.tensor(oof_b, dtype=torch.float32)
    Lx = torch.tensor(oof_x, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)

    torch.manual_seed(seed)
    gate = nn.Sequential(nn.Linear(Gtr.shape[1], 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    idx = torch.arange(len(tr))
    for _ in range(200):
        for b in idx[torch.randperm(len(idx))].split(256):
            opt.zero_grad()
            w = torch.softmax(gate(Gtr[b]), -1)
            lossf(w[:, 0] * Lb[b] + w[:, 1] * Lx[b], yt[b]).backward()
            opt.step()

    # Refit both experts on the full training fold, then gate at test time.
    lb_te = _logit(expert().fit(Xb_tr, y_tr).predict_proba(Xb[te])[:, 1])
    lx_te = _logit(expert().fit(Xx_tr, y_tr).predict_proba(Xx[te])[:, 1])
    with torch.no_grad():
        w = torch.softmax(
            gate(torch.tensor(sc.transform(Xb[te]), dtype=torch.float32)), -1
        ).numpy()
    p = w[:, 0] * lb_te + w[:, 1] * lx_te
    return (p, w[:, 0]) if return_gate else p


# ---------------------------------------------------------------- decision rule
def verdict(a: list, b: list) -> dict:
    """Rule 9, applied verbatim: 2x pooled fold SD AND >=80% sign agreement."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    mean_d = float(d.mean())
    pooled_sd = float(np.std(np.concatenate([a, b])))
    agree = int((np.sign(d) == np.sign(mean_d)).sum()) if mean_d != 0 else 0
    real = abs(mean_d) > 2 * pooled_sd and agree >= 8
    return dict(delta=mean_d, pooled_sd=pooled_sd, sign=f"{agree}/10",
                verdict="REAL" if real else "tie")


# ---------------------------------------------------------------- run
def main() -> None:
    pat = pd.read_csv(REPO / COH / "patients.csv")
    raw = pd.read_csv(REPO / COH / "scans.csv")
    _, Xs, y, Xbase = load(COH)  # rule 2: base table via the sanctioned path

    xmod_df = build_xmodal_block(pat, raw)
    assert len(xmod_df) == len(y), "row count mismatch"
    assert not xmod_df.index.to_series().reset_index(drop=True).ne(
        pat["patient_id"]).any(), "row ORDER mismatch vs load()"
    Xmod = np.nan_to_num(xmod_df.values, nan=0.0, posinf=0.0, neginf=0.0)
    Xaug = np.hstack([Xbase, Xmod])

    print(f"cohort {COH}   n={len(y)}   prevalence={y.mean():.3f}")
    print(f"base features   : {Xbase.shape[1]}  (build_feature_table_fast)")
    print(f"cross-modal add : {Xmod.shape[1]}  (discordance + interactions + composite)")
    print(f"augmented       : {Xaug.shape[1]}")
    print("5x2 repeated stratified CV, identical folds to the published benchmark\n")

    splits = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(Xs, y)
    )

    arms = {}
    for name in learners():
        arms[f"{name}/base"] = ("learner", name, Xbase)
        arms[f"{name}/+xmod"] = ("learner", name, Xaug)
    arms["GUIDE-HYBRID"] = ("hybrid", None, None)

    auroc = {a: [] for a in arms}
    auprc = {a: [] for a in arms}

    for k, (tr, te) in enumerate(splits, 1):
        for arm, (kind, name, X) in arms.items():
            if kind == "learner":
                m = learners()[name]()
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
            else:
                p = guide_hybrid(Xbase, Xmod, y, tr, te, seed=0)
            auroc[arm].append(roc_auc_score(y[te], p))
            auprc[arm].append(average_precision_score(y[te], p))
        print(f"  fold {k:2d}/10 done", flush=True)

    rows = [
        dict(arm=a, auroc=np.mean(auroc[a]), auroc_sd=np.std(auroc[a]),
             auprc=np.mean(auprc[a]))
        for a in arms
    ]
    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    df.to_csv(HERE / "xmodal_features.csv", index=False)

    print("\n" + "=" * 62)
    print(f"{'arm':<16}{'AUROC':>16}{'AUPRC':>9}")
    print("-" * 62)
    for r in df.itertuples():
        print(f"{r.arm:<16}{r.auroc:>8.3f} +/- {r.auroc_sd:.3f}{r.auprc:>9.3f}")
    print("=" * 62)

    print("\nHARNESS SELF-CHECK  (must land on the published benchmark)")
    print(f"  HGB/base here      {np.mean(auroc['HGB/base']):.3f}")
    print("  GBM published      0.832   <- published_families_benchmark.csv")

    print("\nPHASE 1 -- does the cross-modal block help? (read DOWN each column)")
    print(f"{'learner':<12}{'base':>8}{'+xmod':>9}{'delta':>9}{'2*SD':>8}"
          f"{'sign':>7}  verdict")
    print("-" * 62)
    p1 = []
    for name in learners():
        b, a = auroc[f"{name}/base"], auroc[f"{name}/+xmod"]
        v = verdict(a, b)
        p1.append(dict(comparison=f"{name}: +xmod - base", **v))
        print(f"{name:<12}{np.mean(b):>8.3f}{np.mean(a):>9.3f}{v['delta']:>+9.3f}"
              f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    print("\nPHASE 2 -- does splitting the views and gating beat one joint model?")
    print(f"{'':<12}{'joint':>8}{'gated':>9}{'delta':>9}{'2*SD':>8}{'sign':>7}  verdict")
    print("-" * 62)
    j, h = auroc["XGB/+xmod"], auroc["GUIDE-HYBRID"]
    v = verdict(h, j)
    p1.append(dict(comparison="GUIDE-HYBRID - XGB/+xmod", **v))
    print(f"{'XGB views':<12}{np.mean(j):>8.3f}{np.mean(h):>9.3f}{v['delta']:>+9.3f}"
          f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    # ---- POST-HOC, and flagged as such ------------------------------------
    # Not in the pre-specified list above: the pre-registered comparisons were
    # within-learner (Phase 1) and split-vs-joint (Phase 2). This asks the headline
    # question those two imply -- the proposal's FULL stack against the off-the-shelf
    # incumbent -- and is reported as post-hoc because it was added after seeing the
    # table, not before.
    print("\nPOST-HOC -- the proposal's full stack vs the off-the-shelf incumbent")
    print(f"{'':<22}{'CatBoost':>9}{'arm':>7}{'delta':>9}{'2*SD':>8}{'sign':>7}  verdict")
    print("-" * 68)
    post = []
    for arm in ("GUIDE-HYBRID", "LOGREG/+xmod", "HGB/+xmod"):
        v = verdict(auroc[arm], auroc["CatBoost/base"])
        post.append(dict(comparison=f"{arm} - CatBoost/base", post_hoc=True, **v))
        print(f"{arm:<22}{np.mean(auroc['CatBoost/base']):>9.3f}"
              f"{np.mean(auroc[arm]):>7.3f}{v['delta']:>+9.3f}"
              f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    for r in p1:
        r["post_hoc"] = False
    pd.DataFrame(p1 + post).to_csv(HERE / "xmodal_verdicts.csv", index=False)

    # Per-fold values, so any later paired test or forest plot recomputes from source
    # instead of re-running the CV (and re-deriving is how ad-hoc features got in).
    pd.DataFrame(
        [dict(arm=a, fold=i + 1, auroc=auroc[a][i], auprc=auprc[a][i])
         for a in arms for i in range(len(splits))]
    ).to_csv(HERE / "xmodal_per_fold.csv", index=False)

    print("\nwrote xmodal_features.csv, xmodal_verdicts.csv, xmodal_per_fold.csv")


if __name__ == "__main__":
    main()
