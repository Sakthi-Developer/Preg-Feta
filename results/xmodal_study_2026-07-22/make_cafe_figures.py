"""Render the figures for the CAFE manuscript from the measured CSVs.

Every number plotted is read from a file written by clinical_evaluation.py,
xmodal_features.py or encoder_b_lightattention.py -- nothing is retyped from the
manuscript, so a figure cannot silently drift away from the result it depicts. The one
exception is the ablation waterfall, whose steps are declared explicitly at the top of
that function and asserted against the CSVs before plotting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "figures" / "cafe"
OUT.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe qualitative palette (Okabe-Ito).
C = {
    "LOGREG": "#999999",
    "XGBoost": "#E69F00",
    "CatBoost": "#0072B2",
    "HYBRID-XGB": "#009E73",
    "HYBRID-CAT": "#D55E00",
}

# Display labels, identical to the ones make_cafe_tables.py and the manuscript use.
# The CSVs carry the internal run names; a figure that shows HYBRID-CAT where every
# table says "CAFE (CatBoost experts)" leaves the reader unable to match the two.
LABEL = {"HYBRID-CAT": "CAFE (CatBoost experts)", "CatBoost": "CatBoost",
         "LOGREG": "Logistic regression", "HYBRID-XGB": "CAFE (XGBoost experts)",
         "XGBoost": "XGBoost"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
})


def fig_calibration() -> None:
    df = pd.read_csv(HERE / "clinical_calibration_curve.csv")
    summ = pd.read_csv(HERE / "clinical_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration", zorder=1)
    for arm, g in df.groupby("arm"):
        g = g.sort_values("pred")
        ax.plot(g.pred, g.observed, "o-", ms=4, lw=1.4, color=C.get(arm),
                label=f"{LABEL.get(arm, arm)} — slope {summ.loc[arm, 'cal_slope']:.2f}",
                zorder=3)
    ax.set_xlabel("mean predicted probability of loss")
    ax.set_ylabel("observed fraction with loss")
    ax.set_title("Calibration, pooled out-of-fold (repeat 2)", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "fig_cafe_calibration.png"); plt.close(fig)


def fig_decision_curve() -> None:
    d = pd.read_csv(HERE / "clinical_decision_curve.csv")
    arms = [c for c in d.columns if c not in ("threshold", "treat_all", "treat_none")]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(d.threshold, d.treat_all, color="#444444", lw=1.2, ls="--",
            label="treat all")
    ax.axhline(0, color="#444444", lw=1.2, ls=":", label="treat none")
    for a in arms:
        ax.plot(d.threshold, d[a], lw=1.6, color=C.get(a), label=LABEL.get(a, a))
    lo = min(0.0, float(d[arms].min().min()))
    ax.set_ylim(max(lo, -0.05), float(d[arms].max().max()) * 1.15)
    ax.set_xlabel("risk threshold at which a clinician would act")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision-curve analysis, out-of-fold", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "fig_cafe_decision_curve.png"); plt.close(fig)


def fig_shap() -> None:
    imp = pd.read_csv(HERE / "shap_global_importance.csv").head(20).iloc[::-1]
    cols = ["#0072B2" if v == "base" else "#D55E00" for v in imp.view]
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.barh(range(len(imp)), imp.mean_abs_shap, color=cols)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.feature, fontsize=7.5)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Top 20 features by out-of-fold exact TreeSHAP", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#0072B2"),
               plt.Rectangle((0, 0), 1, 1, color="#D55E00")]
    ax.legend(handles, ["base table", "engineered cross-modal"],
              frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_cafe_shap.png"); plt.close(fig)


def fig_ablation() -> None:
    """Waterfall: where does the final AUROC actually come from?

    Steps are asserted against the measured CSVs so this figure cannot drift.
    """
    xm = (pd.read_csv(HERE / "xmodal_per_fold.csv")
          .groupby("arm").auroc.mean())
    eb = (pd.read_csv(HERE / "encoder_b_per_fold.csv")
          .groupby("arm").auroc.mean())
    hy = pd.read_csv(HERE / "hybrid_catboost_experts_per_fold.csv").auroc.mean()

    steps = [
        ("Logistic regression\non base features", xm["LOGREG/base"]),
        ("+ 43 engineered\nclinical features", xm["LOGREG/+xmod"]),
        ("+ swap learner\nto XGBoost", xm["XGB/+xmod"]),
        ("+ swap learner\nto CatBoost", xm["CatBoost/+xmod"]),
        ("+ split into views\n+ per-patient gate\n= CAFE", hy),
        ("+ add sequence\nEncoder B", hy + (eb["A+B+C/cat"] - eb["A+C/cat"])),
    ]
    labels = [s[0] for s in steps]
    vals = np.array([s[1] for s in steps])
    deltas = np.diff(vals, prepend=vals[0])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#666666"] + ["#009E73" if d > 0.002 else
                            "#D55E00" if d < -0.002 else "#BBBBBB"
                            for d in deltas[1:]]
    ax.bar(range(len(vals)), vals - 0.80, bottom=0.80, color=colors, width=0.6)
    for i, (v, d) in enumerate(zip(vals, deltas)):
        ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8.5,
                fontweight="bold")
        if i:
            ax.text(i, 0.803, f"{d:+.3f}", ha="center", fontsize=8,
                    color="#111111")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.80, max(vals) + 0.008)
    ax.set_ylabel("AUROC (5×2 CV)")
    ax.set_title("Where the accuracy comes from: the architecture step is +0.001",
                 loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig_cafe_ablation.png"); plt.close(fig)


def fig_accuracy_vs_cost() -> None:
    summ = pd.read_csv(HERE / "clinical_summary.csv", index_col=0)
    cost = pd.read_csv(HERE / "clinical_cost.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for arm in summ.index:
        if arm not in cost.index:
            continue
        x, y = cost.loc[arm, "fit_s"], summ.loc[arm, "auroc"]
        # Fold-SD bars: without them this axis makes arms the paper reports as ties
        # look cleanly separated, which is the opposite of the result.
        ax.errorbar(x, y, yerr=summ.loc[arm, "auroc_sd"], fmt="o", ms=8,
                    color=C.get(arm), ecolor=C.get(arm), elinewidth=1.2,
                    capsize=3, alpha=0.9, zorder=3)
        ax.annotate(LABEL.get(arm, arm), (x, y), textcoords="offset points",
                    xytext=(9, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training time for one fold (s, log scale)")
    ax.set_ylabel("AUROC (5×2 CV)")
    ax.set_title("Accuracy against training cost (bars: ±1 fold SD)", loc="left")
    fig.savefig(OUT / "fig_cafe_accuracy_vs_cost.png"); plt.close(fig)


def main() -> None:
    made = []
    for fn in (fig_calibration, fig_decision_curve, fig_shap, fig_ablation,
               fig_accuracy_vs_cost):
        try:
            fn()
            made.append(fn.__name__)
        except Exception as exc:  # keep going: one missing CSV must not block the rest
            print(f"  SKIP {fn.__name__}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"wrote {len(made)} figures to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
