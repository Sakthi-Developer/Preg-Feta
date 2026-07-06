#!/usr/bin/env python3
"""Generate the nine manuscript figures as ~200 DPI PNGs into figures/.

All numbers are pulled from the saved artifacts in results/ and data/generated/
(the source of truth). AUROC/AUPRC in legends are recomputed from the saved
per-sample test_predictions.csv files, not from the stale model_comparison.md.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# Colour palette (colour-blind friendly, consistent across figures)
C = {
    "viable": "#2C7FB8",   # blue
    "loss":   "#D95F02",   # orange
    "us":     "#1B9E77",   # ultrasound (green)
    "mat":    "#7570B3",   # maternal (purple)
    "phys":   "#D95F02",   # physiological edges
    "temp":   "#2C7FB8",   # temporal edges
    "m2u":    "#7570B3",   # maternal->us edges
    "proposed": "#B2182B",
    "baseline": "#9E9E9E",
    "box":    "#F2F5F8",
    "boxedge":"#37474F",
}

def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared loaders
# ─────────────────────────────────────────────────────────────────────────────
PRED_FILES = {
    "FETA-Transformer":            ("results/feta_transformer/test_predictions.csv", True),
    "PREG-Net":                    ("results/preg_net/test_predictions.csv", True),
    "Ensemble (avg)":              ("results/ensemble/test_predictions.csv", True),
    "Ensemble (learned)":          ("results/ensemble_learned/test_predictions.csv", True),
    "Logistic regression":         ("results/baselines/sklearn_logistic_regression/test_predictions.csv", False),
    "Random forest":               ("results/baselines/random_forest/test_predictions.csv", False),
    "XGBoost":                     ("results/baselines/xgboost/test_predictions.csv", False),
    "HistGradientBoosting":        ("results/baselines/hist_gradient_boosting/test_predictions.csv", False),
    "MLP":                         ("results/baselines/torch_mlp/test_predictions.csv", False),
}

def load_pred(rel):
    return pd.read_csv(os.path.join(ROOT, rel))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Framework overview (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, fc=C["box"], ec=C["boxedge"], fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2, color=C["boxedge"], style="-|>", lw=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=16, color=color, lw=lw))

    # main horizontal spine y
    ymid = 4.4
    # (a) generation, (b) preprocessing on the spine
    box(0.2, ymid, 2.1, 1.2, "(a) Synthetic cohort\ngeneration\n(growth curves →\npatients.csv, scans.csv)")
    box(2.7, ymid, 2.1, 1.2, "(b) Preprocessing\nstratified split ·\ntrain-only z-score ·\npad/mask T=5")
    # branch point
    box(5.2, ymid, 2.0, 1.2, "(c) Two parallel\nrepresentations")

    # two model branches
    box(7.7, 5.15, 2.1, 0.95, "Scan-sequence tensor\n→ FETA-Transformer",
        fc="#E3F0F7", ec=C["viable"])
    box(7.7, 3.55, 2.1, 0.95, "Knowledge graph\n→ PREG-Net",
        fc="#EFEBF5", ec=C["mat"])

    # fusion
    box(10.1, ymid, 1.7, 1.2, "(d) Late-fusion\nensemble\n(avg / learned)",
        fc="#FFF3E6", ec=C["loss"])

    # bottom row: training + eval
    box(7.7, 1.7, 2.1, 1.0, "(e) Weighted BCE training\nearly stop on val AUROC")
    box(10.1, 1.7, 1.7, 1.0, "(f) Evaluation +\nattention\nexplainability",
        fc="#EAF5EA", ec=C["us"])

    # arrows along spine
    arrow(2.3, ymid + 0.6, 2.7, ymid + 0.6)
    arrow(4.8, ymid + 0.6, 5.2, ymid + 0.6)
    # branch to models
    arrow(7.2, ymid + 0.9, 7.7, 5.62)
    arrow(7.2, ymid + 0.3, 7.7, 4.02)
    # models to fusion
    arrow(9.8, 5.62, 10.55, ymid + 1.2)
    arrow(9.8, 4.02, 10.55, ymid + 1.2)
    # fusion to training/eval
    arrow(10.95, ymid, 10.95, 2.7)
    arrow(9.8, 2.2, 10.1, 2.2)  # training -> eval

    # "graph derived from same tensors" annotation (dashed link between the two
    # model branches, callout placed in the empty lower-centre area)
    ax.add_patch(FancyArrowPatch((8.75, 5.15), (8.75, 4.5),
                 arrowstyle="<->", mutation_scale=12,
                 color=C["mat"], lw=1.3, linestyle=(0, (4, 3))))
    ax.add_patch(FancyArrowPatch((6.2, 3.0), (8.55, 4.55),
                 arrowstyle="-|>", mutation_scale=11,
                 color=C["mat"], lw=1.0, linestyle=(0, (3, 3))))
    ax.text(6.0, 2.7, "the graph is derived from the same temporal/maternal\n"
                      "tensors — not from a separate data source",
            fontsize=8, color=C["mat"], ha="center", va="center", style="italic")

    ax.text(6.0, 6.0, "Proposed framework overview", ha="center",
            fontsize=13, fontweight="bold")
    save(fig, "fig1_framework_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — FETA-Transformer architecture (vertical schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 10))
    ax.set_xlim(0, 7.2); ax.set_ylim(0, 10.4); ax.axis("off")
    cx = 3.6
    w = 4.4

    stages = [
        ("Input:  T×4 scan tensor +\ngestational ages + mask", "#F2F5F8", C["boxedge"]),
        ("Modality-specific projections\n(FHR, CRL, GS, YSD → d_model)", "#E3F0F7", C["viable"]),
        ("Additive continuous\npositional encoding (gest. age)", "#E3F0F7", C["viable"]),
        ("LayerNorm + dropout", "#F2F5F8", C["boxedge"]),
        ("2-layer Transformer encoder\n(4 heads, self-attention)", "#E3F0F7", C["viable"]),
        ("Attention pooling\n(learned query, weights $\\alpha_t$)", "#FFF3E6", C["loss"]),
        ("Maternal cross-attention\n(maternal vector queries pooled rep.)", "#EFEBF5", C["mat"]),
        ("Linear classifier", "#F2F5F8", C["boxedge"]),
        ("Logit", "#EAF5EA", C["us"]),
    ]
    n = len(stages)
    h = 0.82
    gap = (10.0 - 0.3 - n * h) / (n - 1)
    y = 10.0 - h
    centers = []
    for text, fc, ec in stages:
        ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=ec, lw=1.5))
        ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=9)
        centers.append(y + h / 2)
        y -= (h + gap)

    # arrows
    for i in range(n - 1):
        y_top = centers[i] - h / 2
        y_bot = centers[i + 1] + h / 2
        ax.add_patch(FancyArrowPatch((cx, y_top), (cx, y_bot),
                     arrowstyle="-|>", mutation_scale=15, color=C["boxedge"], lw=1.6))

    # explainability callouts on attention stages (indices 5 and 6)
    for idx, label in [(5, "$\\alpha_t$ temporal-pooling\nweights"),
                       (6, "maternal→time\ncross-attention")]:
        yc = centers[idx]
        ax.add_patch(FancyArrowPatch((cx + w / 2, yc), (cx + w / 2 + 0.5, yc),
                     arrowstyle="-|>", mutation_scale=12, color=C["loss"], lw=1.3))
        ax.text(cx + w / 2 + 0.55, yc, label, ha="left", va="center",
                fontsize=8, color=C["loss"], style="italic")

    ax.text(cx, 10.25, "FETA-Transformer architecture", ha="center",
            fontsize=13, fontweight="bold")
    ax.text(6.9, 5.4, "→ explainability module", rotation=90, ha="center",
            va="center", fontsize=8.5, color=C["loss"])
    save(fig, "fig2_feta_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — PREG-Net architecture + 27-node knowledge graph (schematic)
# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    fig, (axg, axp) = plt.subplots(1, 2, figsize=(12.5, 6.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # LEFT: 27-node graph. 20 temporal (5 positions x 4 vars) + 7 maternal
    axg.set_xlim(0, 10); axg.set_ylim(0, 10); axg.axis("off")
    axg.set_title("27-node knowledge graph")

    vars4 = ["FHR", "CRL", "GS", "YSD"]
    mat7 = ["age", "bmi", "parity", "gravidity", "prev_loss", "IVF", "singleton"]
    us_pos = {}
    # 5 scan positions as columns, 4 vars as rows in the upper region
    for t in range(5):
        x = 1.2 + t * 1.9
        for v, var in enumerate(vars4):
            y = 8.6 - v * 0.85
            us_pos[(t, var)] = (x, y)
    mat_pos = {}
    for m, name in enumerate(mat7):
        x = 1.2 + m * 1.28
        mat_pos[name] = (x, 1.4)

    # edges: temporal (same var across adjacent t) - blue
    for var in vars4:
        for t in range(4):
            a = us_pos[(t, var)]; b = us_pos[(t + 1, var)]
            axg.plot([a[0], b[0]], [a[1], b[1]], color=C["temp"], lw=0.9, alpha=0.55, zorder=1)
    # physiological (within a scan, a few clinically-linked pairs) - orange
    phys_pairs = [("FHR", "CRL"), ("CRL", "GS"), ("GS", "YSD"), ("FHR", "YSD")]
    for t in range(5):
        for a, b in phys_pairs:
            pa = us_pos[(t, a)]; pb = us_pos[(t, b)]
            axg.plot([pa[0], pb[0]], [pa[1], pb[1]], color=C["phys"], lw=0.8, alpha=0.5, zorder=1)
    # maternal -> ultrasound (sample a subset to avoid clutter) - purple
    rng = np.random.default_rng(0)
    for name in mat7:
        mp = mat_pos[name]
        targets = [(t, v) for t in range(5) for v in vars4]
        for tt in rng.choice(len(targets), size=3, replace=False):
            up = us_pos[targets[tt]]
            axg.plot([mp[0], up[0]], [mp[1], up[1]], color=C["m2u"], lw=0.5, alpha=0.28, zorder=1)

    # nodes
    for (t, var), (x, y) in us_pos.items():
        axg.add_patch(Circle((x, y), 0.26, fc=C["us"], ec="white", lw=1, zorder=3))
        axg.text(x, y, var, ha="center", va="center", fontsize=6.2, color="white", zorder=4)
    for name, (x, y) in mat_pos.items():
        axg.add_patch(Circle((x, y), 0.30, fc=C["mat"], ec="white", lw=1, zorder=3))
        axg.text(x, y - 0.62, name, ha="center", va="center", fontsize=6.4, zorder=4)

    # scan-position labels
    for t in range(5):
        axg.text(1.2 + t * 1.9, 9.2, f"scan {t+1}", ha="center", fontsize=7.5, color="#555")
    axg.text(0.2, 8.6, "US\nnodes\n(20)", fontsize=7.5, color=C["us"], ha="center", va="center")
    axg.text(0.2, 1.4, "maternal\nnodes (7)", fontsize=7.5, color=C["mat"], ha="center", va="center")

    legend = [
        Line2D([0], [0], color=C["phys"], lw=2, label="physiological"),
        Line2D([0], [0], color=C["temp"], lw=2, label="temporal (same variable)"),
        Line2D([0], [0], color=C["m2u"], lw=2, label="maternal → ultrasound"),
    ]
    axg.legend(handles=legend, loc="lower right", fontsize=7.5, framealpha=0.9)

    # RIGHT: pipeline
    axp.set_xlim(0, 6); axp.set_ylim(0, 10); axp.axis("off")
    axp.set_title("Processing pipeline")
    steps = [
        ("Node features\n(27 nodes × 1)", "#F2F5F8", C["boxedge"]),
        ("Graph-attention layer 1\n(4 heads, per-edge $\\alpha_{vu}$)", "#EFEBF5", C["mat"]),
        ("Graph-attention layer 2\n(4 heads, per-edge $\\alpha_{vu}$)", "#EFEBF5", C["mat"]),
        ("Masked graph readout", "#F2F5F8", C["boxedge"]),
        ("Linear classifier", "#F2F5F8", C["boxedge"]),
        ("Logit", "#EAF5EA", C["us"]),
    ]
    ns = len(steps); h = 0.95
    gap = (9.0 - ns * h) / (ns - 1)
    y = 9.3 - h; centers = []
    for text, fc, ec in steps:
        axp.add_patch(FancyBboxPatch((1.0, y), 4.0, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=ec, lw=1.5))
        axp.text(3.0, y + h / 2, text, ha="center", va="center", fontsize=8.6)
        centers.append(y + h / 2); y -= (h + gap)
    for i in range(ns - 1):
        axp.add_patch(FancyArrowPatch((3.0, centers[i] - h / 2), (3.0, centers[i + 1] + h / 2),
                     arrowstyle="-|>", mutation_scale=14, color=C["boxedge"], lw=1.5))
    # callouts
    axp.add_patch(FancyArrowPatch((5.0, centers[1]), (5.5, centers[1]),
                 arrowstyle="-|>", mutation_scale=11, color=C["loss"], lw=1.2))
    axp.text(5.55, (centers[1] + centers[2]) / 2, "edge-attention\noutput", fontsize=7.5,
             color=C["loss"], ha="left", va="center", style="italic")
    axp.add_patch(FancyArrowPatch((5.0, centers[3]), (5.5, centers[3]),
                 arrowstyle="-|>", mutation_scale=11, color=C["loss"], lw=1.2))
    axp.text(5.55, centers[3], "node-importance\noutput", fontsize=7.5,
             color=C["loss"], ha="left", va="center", style="italic")

    fig.suptitle("PREG-Net architecture and knowledge graph", fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig3_pregnet_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Cohort & trajectories (2x2)
# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    patients = load_pred("data/generated/patients.csv")
    scans = load_pred("data/generated/scans.csv")
    lab = patients.set_index("patient_id")["label"]
    scans = scans.join(lab, on="patient_id")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))

    # (a) class balance
    ax = axes[0, 0]
    vc = patients["label"].value_counts().reindex([0, 1])
    bars = ax.bar(["Viable\n(label 0)", "Loss\n(label 1)"], vc.values,
                  color=[C["viable"], C["loss"]], edgecolor="black", lw=0.6)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v}\n({v/len(patients)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("patients"); ax.set_title("(a) Outcome class balance")
    ax.set_ylim(0, vc.max() * 1.18)

    # (b) scans per patient
    ax = axes[0, 1]
    spp = scans.groupby("patient_id").size()
    bins = np.arange(spp.min() - 0.5, spp.max() + 1.5, 1)
    ax.hist(spp, bins=bins, color="#6BAED6", edgecolor="black", lw=0.6)
    ax.set_xlabel("scans per patient"); ax.set_ylabel("patients")
    ax.set_title("(b) Scans per patient")
    ax.set_xticks(range(int(spp.min()), int(spp.max()) + 1))

    # (c) FHR vs gest week by outcome  (d) CRL
    def traj(ax, col, title, ylab):
        for label, color, name in [(0, C["viable"], "Viable"), (1, C["loss"], "Loss")]:
            sub = scans[scans["label"] == label].copy()
            sub["gw"] = sub["gestational_age_weeks"].round().astype(int)
            g = sub.groupby("gw")[col]
            m = g.mean(); sd = g.std().fillna(0)
            wk = m.index.values
            ax.plot(wk, m.values, color=color, lw=2, marker="o", ms=3.5, label=name)
            ax.fill_between(wk, m - sd, m + sd, color=color, alpha=0.18)
        ax.set_xlabel("gestational age (weeks)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.legend(fontsize=8.5)

    traj(axes[1, 0], "FHR", "(c) Fetal heart rate by outcome", "FHR (bpm)")
    traj(axes[1, 1], "CRL", "(d) Crown-rump length by outcome", "CRL (mm)")

    fig.suptitle("Synthetic cohort and trajectory visualization", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig4_cohort_trajectories.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — ROC + PR curves, 9 models
# ─────────────────────────────────────────────────────────────────────────────
def fig5():
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(13, 6))
    prevalence = None
    # proposed models get distinct bold colours; baselines thin grey-ish
    proposed_colors = {
        "FETA-Transformer": "#B2182B",
        "PREG-Net": "#2166AC",
        "Ensemble (avg)": "#1B7837",
        "Ensemble (learned)": "#762A83",
    }
    baseline_colors = ["#8C8C8C", "#B0B0B0", "#6E6E6E", "#A5A5A5", "#7C7C7C"]
    bi = 0
    # order: proposed first
    for name, (rel, is_prop) in PRED_FILES.items():
        df = load_pred(rel)
        y, p = df["label"].values, df["probability"].values
        if prevalence is None:
            prevalence = y.mean()
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        if is_prop:
            col = proposed_colors[name]; lw = 2.6; z = 5; alpha = 0.95
        else:
            col = baseline_colors[bi % len(baseline_colors)]; bi += 1
            lw = 1.1; z = 2; alpha = 0.8
        axr.plot(fpr, tpr, color=col, lw=lw, zorder=z, alpha=alpha,
                 label=f"{name} (AUROC={au:.3f})")
        axp.plot(rec, prec, color=col, lw=lw, zorder=z, alpha=alpha,
                 label=f"{name} (AUPRC={ap:.3f})")

    axr.plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=1)
    axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
    axr.set_title("ROC curves (test split)")
    axr.set_xlim(-0.01, 1.01); axr.set_ylim(-0.01, 1.01)
    axr.legend(fontsize=7.4, loc="lower right")

    axp.axhline(prevalence, ls="--", color="#BBBBBB", lw=1,
                label=f"prevalence ({prevalence:.3f})")
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    axp.set_title("Precision–recall curves (test split)")
    axp.set_xlim(-0.01, 1.01); axp.set_ylim(0.0, 1.02)
    axp.legend(fontsize=7.4, loc="lower left")

    fig.suptitle("Discrimination on the held-out test split", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig5_roc_pr.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Cross-validation forest plot
# ─────────────────────────────────────────────────────────────────────────────
def fig6():
    cv = load_pred("results/cross_validation/summary.csv")
    cv = cv[cv["split"] == "test"].copy()
    name_map = {
        "feta": ("FETA-Transformer", True),
        "preg": ("PREG-Net", True),
        "ensemble": ("Ensemble (avg)", True),
        "learned_ensemble": ("Ensemble (learned)", True),
        "sklearn_logistic_regression": ("Logistic regression", False),
        "random_forest": ("Random forest", False),
        "xgboost": ("XGBoost", False),
        "hist_gradient_boosting": ("HistGradientBoosting", False),
        "torch_mlp": ("MLP", False),
    }
    cv["disp"] = cv["model"].map(lambda m: name_map[m][0])
    cv["proposed"] = cv["model"].map(lambda m: name_map[m][1])
    cv = cv.sort_values("auroc_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    ys = np.arange(len(cv))
    for y, (_, r) in zip(ys, cv.iterrows()):
        col = C["proposed"] if r["proposed"] else C["baseline"]
        ax.errorbar(r["auroc_mean"], y, xerr=r["auroc_std"], fmt="o",
                    color=col, ecolor=col, elinewidth=1.8, capsize=4,
                    ms=8 if r["proposed"] else 6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.text(r["auroc_mean"], y + 0.22,
                f"{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    best = cv["auroc_mean"].max()
    ax.axvline(best, ls="--", color="#B2182B", lw=1, alpha=0.6,
               label=f"best mean ({best:.3f})")
    ax.set_yticks(ys)
    ax.set_yticklabels(cv["disp"])
    for tick, prop in zip(ax.get_yticklabels(), cv["proposed"]):
        if prop:
            tick.set_fontweight("bold"); tick.set_color(C["proposed"])
    ax.set_xlabel("Cross-validation AUROC (mean ± SD, 5-fold)")
    ax.set_title("Cross-validation forest plot")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C["proposed"],
                      markeredgecolor="black", ms=9, label="proposed"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=C["baseline"],
                      markeredgecolor="black", ms=8, label="baseline")]
    ax.legend(handles=handles + [Line2D([0], [0], ls="--", color="#B2182B", label=f"best mean ({best:.3f})")],
              loc="lower left", fontsize=8.5)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    save(fig, "fig6_cv_forest.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Confusion matrices (2x2 grid) for the 4 proposed models
# ─────────────────────────────────────────────────────────────────────────────
def fig7():
    models = [
        ("FETA-Transformer", "results/feta_transformer/test_predictions.csv"),
        ("PREG-Net", "results/preg_net/test_predictions.csv"),
        ("Ensemble (avg)", "results/ensemble/test_predictions.csv"),
        ("Ensemble (learned)", "results/ensemble_learned/test_predictions.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    for ax, (name, rel) in zip(axes.flat, models):
        df = load_pred(rel)
        pred = (df["probability"].values >= 0.5).astype(int)
        cm = confusion_matrix(df["label"].values, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if val > cm.max() * 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred viable", "Pred loss"])
        ax.set_yticklabels(["True viable", "True loss"])
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(f"{name}\n(FP={fp}, FN={fn})", fontsize=10.5)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
    fig.suptitle("Confusion matrices at threshold 0.5 (test split)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig7_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — FETA temporal attention by gestational week
# ─────────────────────────────────────────────────────────────────────────────
def fig8():
    d = load_pred("results/explainability/feta_attention_by_gestational_week.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for outcome, color, name in [("ongoing", C["viable"], "Viable"),
                                 ("loss", C["loss"], "Loss")]:
        sub = d[d["outcome"] == outcome].sort_values("gestational_week")
        ax1.plot(sub["gestational_week"], sub["mean_temporal_attention"],
                 color=color, lw=2, marker="o", ms=5, label=name)
        ax1.fill_between(sub["gestational_week"],
                         sub["mean_temporal_attention"] - sub["sd_temporal_attention"],
                         sub["mean_temporal_attention"] + sub["sd_temporal_attention"],
                         color=color, alpha=0.13)
        ax2.plot(sub["gestational_week"], sub["mean_maternal_to_temporal_attention"],
                 color=color, lw=2, marker="s", ms=5, label=name)
    ax1.set_xlabel("gestational age (weeks)")
    ax1.set_ylabel("mean temporal-pooling attention $\\alpha_t$")
    ax1.set_title("Temporal-pooling attention")
    ax1.legend(fontsize=9); ax1.grid(ls=":", alpha=0.4)
    ax2.set_xlabel("gestational age (weeks)")
    ax2.set_ylabel("mean maternal→time cross-attention")
    ax2.set_title("Maternal→time cross-attention")
    ax2.legend(fontsize=9); ax2.grid(ls=":", alpha=0.4)
    fig.suptitle("FETA temporal attention by gestational week",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig8_feta_attention.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — PREG-Net node importance + edge attention
# ─────────────────────────────────────────────────────────────────────────────
def fig9():
    node = load_pred("results/explainability/preg_node_importance_summary.csv")
    edge = load_pred("results/explainability/preg_edge_attention_summary.csv")
    fig, (axn, axe) = plt.subplots(1, 2, figsize=(13, 6.4),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # LEFT: node importance bars, coloured by family
    node = node.sort_values("mean_importance")
    colors = [C["us"] if f == "temporal" else C["mat"] for f in node["node_family"]]
    ys = np.arange(len(node))
    axn.barh(ys, node["mean_importance"], xerr=node["sd_importance"],
             color=colors, edgecolor="black", lw=0.5,
             error_kw=dict(elinewidth=1, capsize=2, alpha=0.6))
    axn.set_yticks(ys); axn.set_yticklabels(node["node_feature"], fontsize=8.5)
    axn.set_xlabel("mean node importance")
    axn.set_title("Node importance")
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=C["us"], ms=10, label="ultrasound"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=C["mat"], ms=10, label="maternal")]
    axn.legend(handles=handles, fontsize=9, loc="lower right")
    axn.grid(axis="x", ls=":", alpha=0.4)

    # RIGHT: mean edge attention grouped by edge family
    fam_map = {"physiological": C["phys"], "temporal": C["temp"], "maternal_to_us": C["m2u"]}
    fam_disp = {"physiological": "physiological", "temporal": "temporal",
                "maternal_to_us": "maternal → ultrasound"}
    grp = edge.groupby("edge_type")["mean_attention"].agg(["mean", "std", "count"])
    grp = grp.reindex(["physiological", "temporal", "maternal_to_us"])
    xs = np.arange(len(grp))
    bars = axe.bar(xs, grp["mean"], yerr=grp["std"],
                   color=[fam_map[f] for f in grp.index], edgecolor="black", lw=0.6,
                   error_kw=dict(elinewidth=1.2, capsize=4))
    for b, (idx, r) in zip(bars, grp.iterrows()):
        axe.text(b.get_x() + b.get_width() / 2, r["mean"] + r["std"] + 0.004,
                 f"{r['mean']:.3f}\n(n={int(r['count'])})", ha="center", va="bottom", fontsize=8)
    axe.set_xticks(xs); axe.set_xticklabels([fam_disp[f] for f in grp.index], fontsize=9)
    axe.set_ylabel("mean edge attention $\\alpha_{vu}$")
    axe.set_title("Edge attention by family")
    axe.set_ylim(0, (grp["mean"] + grp["std"]).max() * 1.25)
    axe.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle("PREG-Net node importance and edge attention",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig9_pregnet_explainability.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9()
    print("\nAll figures written to", FIG)
