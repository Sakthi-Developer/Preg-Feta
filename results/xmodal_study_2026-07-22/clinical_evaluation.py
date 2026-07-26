"""Phase 3 + Phase 4 of the guide's checklist: clinical evaluation and interpretability.

The accuracy question is already settled elsewhere in this study (every arm ties). What
was never measured is whether the arms differ on the axes a clinician actually cares
about: are the probabilities trustworthy, does using the model beat treating everyone,
and can a decision be explained to the patient in front of you. Those are separate
properties from discrimination -- two models with identical AUROC can differ enormously
in calibration -- so a tie in AUROC does NOT imply a tie here. This measures it.

Arms are compared on IDENTICAL 5x2 folds (RepeatedStratifiedKFold(5, 2, random_state=0)),
the same folds as the eight-family benchmark and the cross-modal study, so every number
here is directly comparable to every number already published from this cohort.

INPUT PARITY (rule 1). Every arm sees the SAME union view -- the sanctioned base feature
table plus the 43 cross-modal features. The hybrid splits that union into two views and
gates them; the baselines get the union whole. Nobody is starved, so a hybrid win could
not be an information artefact, and a hybrid loss cannot be excused as one.

RULE 2. The base table is reached through published_families_benchmark.load(), never
rebuilt ad hoc -- ad-hoc features understate the tree by ~0.019 and have manufactured a
false win twice in this project's history.

THRESHOLD SELECTION. Repeated CV gives two complete out-of-fold prediction vectors (one
per repeat). The screening threshold is chosen on repeat 1 and evaluated on repeat 2, so
the operating point is never selected and scored on the same predictions. Picking the
threshold on the same data you report it from inflates sensitivity by construction.

SHAP is exact TreeSHAP from the boosting libraries themselves (CatBoost ShapValues,
XGBoost pred_contribs) -- not the `shap` package, which is not a dependency here. Values
are out-of-fold: every patient is explained by a model that never saw them in training.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE), str(REPO / "results" / "regime_study_2026-07-19")):
    sys.path.insert(0, p)

from catboost import CatBoostClassifier  # noqa: E402
from published_families_benchmark import COH, load  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from scripts.scaling_sweep import build_feature_table_fast  # noqa: E402
from xmodal_features import SEED, build_xmodal_block, guide_hybrid, verdict  # noqa: E402

# Screening target. Early pregnancy loss is the event we must not miss, so the operating
# point is set by sensitivity and specificity is whatever it turns out to be.
TARGET_SENS = 0.90
# Threshold grid for decision-curve analysis. Beyond ~0.5 the curves are all flat at zero
# at this prevalence, so the grid stops where the clinical question stops being live.
DCA_GRID = np.round(np.arange(0.05, 0.51, 0.01), 2)


# ------------------------------------------------------------------ metrics
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better; scale depends on
    prevalence, so it is only comparable between arms on the SAME cohort."""
    return float(np.mean((p - y) ** 2))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox calibration: regress the outcome on the predicted log-odds.

    slope 1.0 = risks are correctly spread; < 1 = over-confident (extreme risks too
    extreme); > 1 = under-confident. intercept 0 = no systematic over/under-prediction.
    """
    lr = LogisticRegression(max_iter=2000).fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed| over equal-width bins,
    weighted by bin occupancy."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    """Vickers' net benefit at threshold probability pt.

    NB = TP/n - FP/n * pt/(1-pt). Reads as "true positives gained per patient, after
    charging each false positive at the odds the clinician implicitly accepts by
    choosing pt". Directly comparable to treat-all and treat-none on the same axis.
    """
    flag = p >= pt
    tp = float(np.sum(flag & (y == 1)))
    fp = float(np.sum(flag & (y == 0)))
    return tp / len(y) - (fp / len(y)) * (pt / (1.0 - pt))


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Lowest-specificity-cost threshold that still reaches `target` sensitivity."""
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    n_pos = max(int(ys.sum()), 1)
    hit = np.cumsum(ys) / n_pos
    reach = np.nonzero(hit >= target)[0]
    return float(ps[reach[0]]) if len(reach) else float(ps[-1])


def operating_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    flag = p >= thr
    tp = float(np.sum(flag & (y == 1)))
    fp = float(np.sum(flag & (y == 0)))
    fn = float(np.sum(~flag & (y == 1)))
    tn = float(np.sum(~flag & (y == 0)))
    return dict(
        threshold=thr,
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        ppv=tp / max(tp + fp, 1),
        npv=tn / max(tn + fn, 1),
        flagged_frac=float(flag.mean()),
    )


# ------------------------------------------------------------------ arms
def cat() -> CatBoostClassifier:
    return CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False)


def xgb() -> XGBClassifier:
    return XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0)


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main() -> None:
    pat = pd.read_csv(REPO / COH / "patients.csv")
    raw = pd.read_csv(REPO / COH / "scans.csv")
    _, Xs, y, Xbase = load(COH)  # rule 2: sanctioned feature path

    xmod_df = build_xmodal_block(pat, raw)
    Xmod = np.nan_to_num(xmod_df.values, nan=0.0, posinf=0.0, neginf=0.0)
    Xuni = np.column_stack([Xbase, Xmod])

    base_names = [
        c for c in build_feature_table_fast(pat, raw).columns
        if c not in ("patient_id", "label")
    ]
    uni_names = base_names + list(xmod_df.columns)
    assert len(uni_names) == Xuni.shape[1], "feature-name/matrix width mismatch"

    splits = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(Xs, y)
    )
    print(f"cohort {COH}   n={len(y)}   prevalence={y.mean():.3f}")
    print(f"views: base={Xbase.shape[1]}  cross-modal={Xmod.shape[1]}  "
          f"union={Xuni.shape[1]}")
    print("5x2 CV, folds identical to the published benchmark\n")

    # Every arm returns a PROBABILITY. The hybrid's gate emits a convex combination of
    # two logits (softmax weights sum to 1), so a sigmoid of it is a genuine pooled
    # probability rather than an arbitrary score -- which is what makes it calibratable.
    def run_plain(factory, tr, te):
        return factory().fit(Xuni[tr], y[tr]).predict_proba(Xuni[te])[:, 1]

    def run_hybrid(expert, tr, te):
        lg, w0 = guide_hybrid(Xbase, Xmod, y, tr, te, seed=0, expert=expert,
                              return_gate=True)
        return 1.0 / (1.0 + np.exp(-lg)), w0

    arms = {
        "LOGREG": lambda tr, te: (run_plain(logreg, tr, te), None),
        "XGBoost": lambda tr, te: (run_plain(xgb, tr, te), None),
        "CatBoost": lambda tr, te: (run_plain(cat, tr, te), None),
        "HYBRID-XGB": lambda tr, te: run_hybrid(xgb, tr, te),
        "HYBRID-CAT": lambda tr, te: run_hybrid(cat, tr, te),
    }

    rows: list[dict] = []
    # oof[arm][repeat] -> full-length prediction vector; each repeat covers every patient
    # exactly once, so a repeat is a complete out-of-fold partition.
    oof = {a: [np.zeros(len(y)), np.zeros(len(y))] for a in arms}
    oof_gate = [np.zeros(len(y)), np.zeros(len(y))]
    per_fold_path = HERE / "clinical_per_fold.csv"

    for k, (tr, te) in enumerate(splits):
        rep = k // 5
        t_fold = time.time()
        for name, fn in arms.items():
            p, w0 = fn(tr, te)
            oof[name][rep][te] = p
            if name == "HYBRID-CAT" and w0 is not None:
                oof_gate[rep][te] = w0
            slope, intercept = calibration_slope_intercept(y[te], p)
            rows.append(dict(
                arm=name, fold=k + 1, repeat=rep + 1,
                auroc=roc_auc_score(y[te], p),
                auprc=average_precision_score(y[te], p),
                brier=brier(y[te], p),
                cal_slope=slope, cal_intercept=intercept,
                ece=ece(y[te], p),
            ))
        # Persist after every fold: a mid-run death must not cost the whole run again.
        pd.DataFrame(rows).to_csv(per_fold_path, index=False)
        print(f"  fold {k + 1:2d}/10 done  [{time.time() - t_fold:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("arm").agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"),
        brier=("brier", "mean"), brier_sd=("brier", "std"),
        cal_slope=("cal_slope", "mean"), cal_intercept=("cal_intercept", "mean"),
        ece=("ece", "mean"),
    ).sort_values("auroc", ascending=False)

    print("\n" + "=" * 88)
    print("PHASE 3  discrimination and calibration (5x2 CV, mean over 10 folds)")
    print("-" * 88)
    print(f"{'arm':<14}{'AUROC':>16}{'AUPRC':>8}{'Brier':>9}{'slope':>8}"
          f"{'intcpt':>8}{'ECE':>8}")
    for r in agg.itertuples():
        print(f"{r.Index:<14}{r.auroc:>8.3f} +/-{r.auroc_sd:.3f}{r.auprc:>8.3f}"
              f"{r.brier:>9.4f}{r.cal_slope:>8.3f}{r.cal_intercept:>8.3f}{r.ece:>8.4f}")
    print("=" * 88)

    # ---------------- decision rule applied to Brier, not just AUROC ----------------
    print("\nDoes any arm actually differ? (rule 9: |delta| > 2x pooled SD AND sign >=8/10)")
    print(f"{'comparison':<44}{'delta':>9}{'2*SD':>8}{'sign':>7}  verdict")
    print("-" * 88)
    vrows = []
    ref = "CatBoost"
    for metric in ("auroc", "brier"):
        for arm in agg.index:
            if arm == ref:
                continue
            a = df[df.arm == arm].sort_values("fold")[metric].values
            b = df[df.arm == ref].sort_values("fold")[metric].values
            v = verdict(a, b)
            label = f"{metric.upper()}: {arm} - {ref}"
            vrows.append(dict(metric=metric, comparison=f"{arm} - {ref}", **v))
            print(f"{label:<44}{v['delta']:>+9.4f}{2 * v['pooled_sd']:>8.4f}"
                  f"{v['sign']:>7}  {v['verdict']}")
    pd.DataFrame(vrows).to_csv(HERE / "clinical_verdicts.csv", index=False)

    # ---------------- calibration curve (pooled out-of-fold, repeat 2) ----------------
    cal_rows = []
    for arm in agg.index:
        p = oof[arm][1]
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        for b in range(10):
            m = idx == b
            if m.sum() < 10:
                continue
            cal_rows.append(dict(arm=arm, bin=b, bin_mid=(edges[b] + edges[b + 1]) / 2,
                                 n=int(m.sum()), pred=float(p[m].mean()),
                                 observed=float(y[m].mean())))
    pd.DataFrame(cal_rows).to_csv(HERE / "clinical_calibration_curve.csv", index=False)

    # ---------------- screening operating point: pick on rep 1, score on rep 2 -------
    print(f"\nPHASE 3  screening operating point (threshold set on repeat 1 at "
          f"sens>={TARGET_SENS:.2f}, scored on repeat 2)")
    print(f"{'arm':<14}{'thr':>7}{'sens':>8}{'spec':>8}{'PPV':>8}{'NPV':>8}{'flagged':>9}")
    print("-" * 88)
    op_rows = []
    for arm in agg.index:
        thr = threshold_for_sensitivity(y, oof[arm][0], TARGET_SENS)
        m = operating_metrics(y, oof[arm][1], thr)
        op_rows.append(dict(arm=arm, **m))
        print(f"{arm:<14}{thr:>7.3f}{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
              f"{m['ppv']:>8.3f}{m['npv']:>8.3f}{m['flagged_frac']:>9.3f}")
    pd.DataFrame(op_rows).to_csv(HERE / "clinical_operating_point.csv", index=False)

    # ---------------- decision curve analysis ----------------
    prev = float(y.mean())
    dca_rows = []
    for pt in DCA_GRID:
        treat_all = prev - (1 - prev) * (pt / (1 - pt))
        row = dict(threshold=float(pt), treat_all=treat_all, treat_none=0.0)
        for arm in agg.index:
            row[arm] = net_benefit(y, oof[arm][1], float(pt))
        dca_rows.append(row)
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(HERE / "clinical_decision_curve.csv", index=False)

    print("\nPHASE 3  decision curve analysis (net benefit, repeat-2 out-of-fold)")
    print(f"{'thr':>6}{'treat-all':>11}", end="")
    for arm in agg.index:
        print(f"{arm:>12}", end="")
    print()
    print("-" * 88)
    for pt in (0.10, 0.20, 0.30, 0.40, 0.50):
        r = dca[np.isclose(dca.threshold, pt)].iloc[0]
        print(f"{pt:>6.2f}{r.treat_all:>11.4f}", end="")
        for arm in agg.index:
            print(f"{r[arm]:>12.4f}", end="")
        print()

    best_arm = agg.index[0]
    gain = {}
    for arm in agg.index:
        g = [dca[np.isclose(dca.threshold, pt)].iloc[0][arm]
             - max(dca[np.isclose(dca.threshold, pt)].iloc[0].treat_all, 0.0)
             for pt in DCA_GRID]
        gain[arm] = float(np.mean(g))
    print("\nmean net benefit above the better of treat-all/treat-none, over the grid:")
    for arm, g in sorted(gain.items(), key=lambda kv: -kv[1]):
        print(f"  {arm:<14}{g:>+9.4f}")

    # ---------------- PHASE 4: interpretability ----------------
    print("\n" + "=" * 88)
    print("PHASE 4  interpretability")
    print("-" * 88)

    # Out-of-fold exact TreeSHAP: each patient explained by a model that never saw them.
    shap_mat = np.zeros((len(y), Xuni.shape[1]))
    for tr, te in splits[:5]:  # repeat 1 = one complete partition
        m = cat().fit(Xuni[tr], y[tr])
        sv = m.get_feature_importance(type="ShapValues",
                                      data=__import__("catboost").Pool(Xuni[te], y[te]))
        shap_mat[te] = sv[:, :-1]  # last column is the expected-value bias term
    np.save(HERE / "shap_values_oof.npy", shap_mat)

    imp = pd.DataFrame(dict(feature=uni_names,
                            mean_abs_shap=np.abs(shap_mat).mean(0)))
    imp["view"] = ["base" if f in base_names else "cross-modal" for f in imp.feature]
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(HERE / "shap_global_importance.csv", index=False)

    print("global feature importance -- top 15 (exact TreeSHAP, out-of-fold, CatBoost)")
    print(f"{'rank':>5}  {'feature':<28}{'mean|SHAP|':>12}  view")
    for i, r in imp.head(15).iterrows():
        print(f"{i + 1:>5}  {r.feature:<28}{r.mean_abs_shap:>12.4f}  {r.view}")

    tot = imp.groupby("view").mean_abs_shap.sum()
    share = tot / tot.sum()
    print(f"\nshare of total |SHAP| mass: base {share.get('base', 0):.3f}   "
          f"cross-modal {share.get('cross-modal', 0):.3f}")
    print(f"cross-modal features in the top 20: "
          f"{int((imp.head(20).view == 'cross-modal').sum())}/20")

    # Gate behaviour: is it per-patient, and what is it responding to?
    g2 = oof_gate[1]
    print(f"\ngate weight on Encoder A (flat view), out-of-fold repeat 2")
    print(f"  mean {g2.mean():.3f}   SD {g2.std():.3f}   "
          f"min {g2.min():.3f}   max {g2.max():.3f}")
    print(f"  patients routed mostly to A (w>0.5): {float((g2 > 0.5).mean()):.3f}")
    print(f"  patients routed mostly to C (w<0.5): {float((g2 < 0.5).mean()):.3f}")

    gcorr = (pd.DataFrame(Xuni, columns=uni_names)
             .apply(lambda c: np.corrcoef(c, g2)[0, 1])
             .replace([np.inf, -np.inf], np.nan).dropna()
             .sort_values(key=np.abs, ascending=False))
    print("\n  what the gate responds to (|corr| with gate weight, top 8):")
    for f, v in gcorr.head(8).items():
        print(f"    {f:<30}{v:>+7.3f}")
    pd.DataFrame(dict(feature=gcorr.index, corr_with_gate=gcorr.values)).to_csv(
        HERE / "gate_correlates.csv", index=False)

    # Case examples: highest- and lowest-risk patients, with their top SHAP drivers.
    p_ref = oof["HYBRID-CAT"][1]
    order = np.argsort(-p_ref)
    cases = []
    for tag, sel in (("highest-risk", order[:3]), ("lowest-risk", order[-3:])):
        for i in sel:
            top = np.argsort(-np.abs(shap_mat[i]))[:5]
            cases.append(dict(
                case=tag, patient_id=pat.patient_id.iloc[i], label=int(y[i]),
                risk=float(p_ref[i]), gate_A=float(g2[i]),
                drivers="; ".join(f"{uni_names[j]}={Xuni[i, j]:.2f}"
                                  f"(SHAP{shap_mat[i, j]:+.2f})" for j in top),
            ))
    pd.DataFrame(cases).to_csv(HERE / "case_examples.csv", index=False)
    print("\ncase examples (HYBRID-CAT out-of-fold risk, top-5 SHAP drivers each)")
    for c in cases:
        print(f"  {c['case']:<13}{c['patient_id']}  risk={c['risk']:.3f}  "
              f"label={c['label']}  gate_A={c['gate_A']:.2f}")
        print(f"      {c['drivers']}")

    # ---------------- cost: fit and inference ----------------
    tr, te = splits[0]
    cost = []
    for name, factory in (("LOGREG", logreg), ("XGBoost", xgb), ("CatBoost", cat)):
        t0 = time.time()
        m = factory().fit(Xuni[tr], y[tr])
        fit_s = time.time() - t0
        t0 = time.time()
        m.predict_proba(Xuni[te])
        pred_ms = (time.time() - t0) * 1000
        cost.append(dict(model=name, fit_s=fit_s, pred_ms=pred_ms,
                         per_patient_ms=pred_ms / len(te)))
    for name, ex in (("HYBRID-XGB", xgb), ("HYBRID-CAT", cat)):
        t0 = time.time()
        guide_hybrid(Xbase, Xmod, y, tr, te, seed=0, expert=ex)
        total = time.time() - t0
        cost.append(dict(model=name, fit_s=total, pred_ms=np.nan,
                         per_patient_ms=np.nan))
    cdf = pd.DataFrame(cost)
    cdf.to_csv(HERE / "clinical_cost.csv", index=False)
    print("\nPHASE 3  cost on this feature table (single fold, this machine)")
    print(cdf.round(4).to_string(index=False))

    summary = dict(
        cohort=COH, n=int(len(y)), prevalence=float(y.mean()),
        n_features_union=int(Xuni.shape[1]),
        arms={a: dict(auroc=float(agg.loc[a, "auroc"]),
                      auroc_sd=float(agg.loc[a, "auroc_sd"]),
                      auprc=float(agg.loc[a, "auprc"]),
                      brier=float(agg.loc[a, "brier"]),
                      cal_slope=float(agg.loc[a, "cal_slope"]),
                      cal_intercept=float(agg.loc[a, "cal_intercept"]),
                      ece=float(agg.loc[a, "ece"]),
                      mean_nb_gain=gain[a]) for a in agg.index},
        shap_view_share={k: float(v) for k, v in share.items()},
        gate=dict(mean=float(g2.mean()), sd=float(g2.std())),
    )
    (HERE / "clinical_summary.json").write_text(json.dumps(summary, indent=2))
    agg.round(4).to_csv(HERE / "clinical_summary.csv")
    print("\nwrote clinical_per_fold.csv, clinical_verdicts.csv, "
          "clinical_calibration_curve.csv,")
    print("      clinical_operating_point.csv, clinical_decision_curve.csv, "
          "shap_global_importance.csv,")
    print("      gate_correlates.csv, case_examples.csv, clinical_cost.csv, "
          "clinical_summary.json")


if __name__ == "__main__":
    main()
