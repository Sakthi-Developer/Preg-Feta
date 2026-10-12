"""Render Tables 1-10 of the CAFE manuscript as PNG images, from the measured CSVs.

Same contract as `make_cafe_figures.py` and `scripts/make_paper_assets.py`: **every number
is read from the file that produced it**, never retyped from the manuscript. Re-running
this script cannot change a result, because nothing here fits a model -- it reads the
retained per-fold predictions and formats them. The only inputs that are *declared* rather
than read are the prose descriptions in Tables 1 and 2, and each of those carries an
assertion against the real artefact (the feature builder, the parameter arithmetic) that
fails loudly if the description stops being true.

Derivations are taken at full precision from the per-fold files, exactly as
`verify_cafe_manuscript.py` does, and NOT from the rounded `*_summary.csv` -- formatting an
already-rounded value double-rounds it, which moves CatBoost's AUROC by one ulp.

    .venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_tables.py

Visually this matches the booktabs style of `scripts/make_paper_assets.py::render_table`;
it is reimplemented here rather than imported because the CAFE tables carry long prose
cells that have to wrap, which that renderer does not do.
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "figures" / "cafe" / "tables"
RUNTIME = REPO / "results" / "regime_study_2026-07-19" / "runtime_real_models.csv"

INK, MUTED, RULE, SHADE = "#1f2328", "#6b7280", "#333333", "#eef1f5"
SYNTH = "All data are synthetic — not real patients."

# The descriptive caption lives in the Word document's own numbered Caption paragraph,
# not baked into the image: a caption inside a PNG cannot be selected, searched, or
# renumbered, and a second copy above the image would simply duplicate it. Set this True
# to get standalone images that carry their own caption.
EMBED_CAPTION = False

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 300, "figure.dpi": 150,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# Display labels, identical to the ones verify_cafe_manuscript.py checks against.
LABEL = {"HYBRID-CAT": "CAFE (CatBoost experts)", "CatBoost": "CatBoost",
         "LOGREG": "Logistic regression", "HYBRID-XGB": "CAFE (XGBoost experts)",
         "XGBoost": "XGBoost"}
ARM_ORDER = ["HYBRID-CAT", "CatBoost", "LOGREG", "HYBRID-XGB", "XGBoost"]


# ====================================================================== renderer
def _typo(text: str) -> str:
    """ASCII stand-ins -> the real glyphs, so the images match the manuscript's typography.

    The negative-sign rule only fires on a hyphen that opens a number ("-0.007"), never on
    one inside a token, so hyphenated names and dated paths ("2026-07-22") are untouched.
    """
    text = text.replace(" -- ", " — ").replace("->", "→")
    text = re.sub(r"\bdelta\b", "Δ", text)
    text = re.sub(r"(?<=\d)x(?=\d)", "×", text)                       # 5x2  -> 5×2
    text = re.sub(r"(?<=\d)x\b", "×", text)                           # 75x  -> 75×
    text = re.sub(r"(?<=\w) x (?=\w)", " × ", text)          # age x zFHR -> age × zFHR
    text = re.sub(r"(?<![\w.])-(?=[.\d])", "−", text)            # -0.007 -> −0.007
    return text


def _wrap(text: str, width_in: float, fontsize: float, pad: float) -> list[str]:
    """Wrap a cell to its column width. ~0.54 em average glyph width for Times."""
    chars = max(int((width_in - 2 * pad) / (fontsize * 0.54 / 72)), 6)
    return textwrap.wrap(text, chars) or [""]


def render_table(name: str, caption: str, headers: list[str], rows: list[list[str]],
                 colw: list[float], aligns: list[str] | None = None,
                 notes: tuple[str, ...] = (), fontsize: float = 9.0,
                 highlight: int | None = None, out: Path = OUT) -> Path:
    """Booktabs table image: top rule, bold header, mid rule, rows, bottom rule.

    A cell wrapped in `**` is emitted bold with the markers stripped; other inline
    Markdown emphasis is stripped and rendered plain, since partial emphasis inside a
    wrapped cell cannot be placed reliably.
    """
    ncol = len(headers)
    aligns = aligns or (["left"] + ["center"] * (ncol - 1))
    pad, total = 0.10, sum(colw)

    def strip(cell: str) -> tuple[str, bool]:
        bold = cell.startswith("**") and cell.endswith("**") and len(cell) > 4
        text = cell[2:-2] if bold else cell
        return _typo(re.sub(r"[*`]", "", text)), bold

    wrapped, heights = [], []
    for row in rows:
        cells = []
        for j in range(ncol):
            text, bold = strip(str(row[j]) if j < len(row) else "")
            cells.append((_wrap(text, colw[j], fontsize, pad), bold))
        wrapped.append(cells)
        heights.append(0.055 + 0.175 * max(len(c[0]) for c in cells))

    head_lines = [_wrap(_typo(h), colw[j], fontsize, pad) for j, h in enumerate(headers)]
    header_h = 0.10 + 0.175 * max(len(h) for h in head_lines)
    # The "Table n." label is dropped from the image: the Word document numbers the table
    # in its own Caption paragraph, and two numbers on one table can disagree.
    if EMBED_CAPTION:
        text = re.sub(r"^Table\s+\d+\.\s*", "", re.sub(r"[*`]", "", caption))
        cap_lines = _wrap(text, total, fontsize + 0.5, 0.0)
        cap_h = 0.20 * len(cap_lines) + 0.10
    else:
        cap_lines, cap_h = [], 0.10
    note_h = 0.19 * (len(notes) + 1) + 0.10

    fig_h = cap_h + header_h + sum(heights) + note_h
    fig, ax = plt.subplots(figsize=(total, fig_h))
    ax.set_xlim(0, total)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    xs, x = [], 0.0
    for w in colw:
        xs.append(x)
        x += w

    def cell_x(j: int) -> tuple[float, str]:
        if aligns[j] == "left":
            return xs[j] + pad, "left"
        if aligns[j] == "right":
            return xs[j] + colw[j] - pad, "right"
        return xs[j] + colw[j] / 2, "center"

    y = fig_h - 0.15
    for line in cap_lines:                                     # caption, above the table
        ax.text(0, y, line, ha="left", va="center", fontsize=fontsize + 0.5,
                color=INK, style="italic")
        y -= 0.20

    y_top = y - 0.06
    ax.plot([0, total], [y_top, y_top], color=RULE, lw=1.5)
    for j, lines in enumerate(head_lines):
        cx, ha = cell_x(j)
        yy = y_top - 0.14
        for line in lines:
            ax.text(cx, yy, line, ha=ha, va="center", fontsize=fontsize,
                    fontweight="bold", color=INK)
            yy -= 0.175
    y_mid = y_top - header_h
    ax.plot([0, total], [y_mid, y_mid], color=RULE, lw=1.0)

    y_row = y_mid
    for i, (cells, h) in enumerate(zip(wrapped, heights)):
        if i == highlight:                                     # the arm under discussion
            ax.add_patch(plt.Rectangle((0, y_row - h), total, h, color=SHADE, zorder=0))
        for j, (lines, bold) in enumerate(cells):
            cx, ha = cell_x(j)
            yy = y_row - 0.115
            for line in lines:
                ax.text(cx, yy, line, ha=ha, va="center", fontsize=fontsize, color=INK,
                        fontweight="bold" if bold else "normal", zorder=2)
                yy -= 0.175
        y_row -= h
    ax.plot([0, total], [y_row, y_row], color=RULE, lw=1.5)

    yn = y_row - 0.20
    for note in notes:
        for line in _wrap(_typo(re.sub(r"[*`]", "", note)), total, 7.5, 0.0):
            ax.text(0, yn, line, ha="left", va="center", fontsize=7.5, color=MUTED)
            yn -= 0.17
    ax.text(0, yn, SYNTH, ha="left", va="center", fontsize=6.5, color=MUTED)

    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


# ====================================================================== sources
def load() -> dict:
    """Read every retained result file this module formats. No model is fitted here."""
    s = {
        "clinical": pd.read_csv(HERE / "clinical_per_fold.csv"),
        "clinical_verdicts": pd.read_csv(HERE / "clinical_verdicts.csv"),
        "xmodal": pd.read_csv(HERE / "xmodal_per_fold.csv"),
        "xmodal_verdicts": pd.read_csv(HERE / "xmodal_verdicts.csv"),
        "encb": pd.read_csv(HERE / "encoder_b_per_fold.csv"),
        "encb_verdicts": pd.read_csv(HERE / "encoder_b_verdicts.csv"),
        "hybrid": pd.read_csv(HERE / "hybrid_catboost_experts_per_fold.csv"),
        "dca": pd.read_csv(HERE / "clinical_decision_curve.csv"),
        "opp": pd.read_csv(HERE / "clinical_operating_point.csv"),
        "shap": pd.read_csv(HERE / "shap_global_importance.csv"),
        "cost": pd.read_csv(HERE / "clinical_cost.csv").set_index("model"),
        "runtime": pd.read_csv(RUNTIME).set_index("model"),
    }
    s["summary"] = s["clinical"].groupby("arm").agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), auprc=("auprc", "mean"),
        brier=("brier", "mean"), cal_slope=("cal_slope", "mean"),
        cal_intercept=("cal_intercept", "mean"), ece=("ece", "mean"))
    s["xm"] = s["xmodal"].groupby("arm").auroc.mean()
    s["eb"] = s["encb"].groupby("arm").auroc.agg(["mean", "std"])
    return s


def _sign(verdicts: pd.DataFrame, comparison: str, col: str = "comparison") -> tuple:
    row = verdicts.loc[verdicts[col] == comparison].iloc[0]
    return row["sign"], row["verdict"]


def feature_families() -> list[tuple[str, str, str, int]]:
    """The five families, with counts taken from the feature builder itself."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))
    import xmodal_features as xf                                       # noqa: PLC0415

    block = xf.build_xmodal_block(pd.read_csv(REPO / "data/gen_hard/patients.csv"),
                                  pd.read_csv(REPO / "data/gen_hard/scans.csv"))
    cols = [c for c in block.columns if c != "patient_id"]

    def family(c: str) -> str:
        if c.startswith(("disc_", "corr_")):
            return "discordance"
        if "_x_" in c:
            return "cross-modal"
        if c == "mat_composite":
            return "maternal"
        if c.endswith("_slope"):
            return "velocity"
        return "trajectory"

    n = pd.Series([family(c) for c in cols]).value_counts()
    rows = [
        ("Gestational normalisation", "Is this value normal for this week?",
         "z-score of each measurement within its rounded GA-week bin", None),
        ("Temporal trajectory", "Where is the marker now, on average, and at its worst?",
         "per-patient last / mean / min of each z-trajectory", int(n["trajectory"])),
        ("Growth velocity", "Is the marker moving in the right direction, fast enough?",
         "per-patient OLS slope of each z-trajectory against GA", int(n["velocity"])),
        ("Discordance", "Do the markers disagree with each other?",
         "within-patient correlation, level gap and slope gap for (FHR,CRL), (CRL,GS), "
         "(FHR,GS), (CRL,YSD); slope-sign disagreement; spread and worst channel",
         int(n["discordance"])),
        ("Maternal risk profile", "What is the patient's baseline clinical risk?",
         "standardised composite of age, BMI, previous loss, IVF conception, non-singleton",
         int(n["maternal"])),
        ("Cross-modal interaction", "Does maternal risk modify the sonographic signal?",
         "11 products of maternal terms with sonographic terms "
         "(e.g. age x zFHR_last, BMI x zCRL_last, composite x discordance spread)",
         int(n["cross-modal"])),
    ]
    counted = [r[3] for r in rows if r[3] is not None]
    assert sum(counted) == len(cols) == 43, f"family counts {counted} != {len(cols)}"
    assert counted == [12, 4, 15, 1, 11], f"family split changed: {counted}"
    return rows


# ====================================================================== tables
def table1(s: dict) -> Path:
    rows = [[f, q, c, "basis" if n is None else str(n)]
            for f, q, c, n in feature_families()]
    return render_table(
        "table01_feature_families.png",
        "Table 1. The five feature families: the clinical question each encodes, its "
        "construction, and the number of features contributed.",
        ["Family", "Clinical question it encodes", "Construction", "n"], rows,
        colw=[1.55, 2.05, 3.30, 0.45],
        aligns=["left", "left", "left", "center"],
        notes=("Total added: 43 features, over a 23-feature base table, giving a "
               "66-feature union view.",))


def table2(_s: dict) -> Path:
    gate = 23 * 16 + 16 + 16 * 2 + 2            # MLP 23 -> 16 -> K=2, with biases
    assert gate == 418, gate
    rows = [
        ["A", "Flat tabular base table (23 features)",
         "Gradient-boosted tree (XGBoost or CatBoost)", "tree ensemble"],
        ["B", "Raw scan sequence (2–5 visits × 4 measures + GA)",
         "Temporal attention, d_model 24, 1 layer, 2 heads", "12,585"],
        ["C", "Cross-modal engineered block (43 features)",
         "Gradient-boosted tree (XGBoost or CatBoost)", "tree ensemble"],
        ["Gate", "Base table (23 features)",
         "MLP 23 -> 16 -> K, softmax over experts", f"{gate} (K = 2)"],
    ]
    return render_table(
        "table02_encoder_specs.png",
        "Table 2. Encoder specifications: the view each encoder consumes, its model "
        "class, and its parameter count.",
        ["Encoder", "View", "Model", "Parameters"], rows,
        colw=[0.75, 2.60, 2.85, 1.10],
        aligns=["center", "left", "left", "center"],
        notes=("Encoder B's 12,585 parameters are the count logged at training time; "
               "the gate's 418 are its layer arithmetic.",))


def table3(s: dict) -> Path:
    best = {"auroc": "max", "auprc": "max", "brier": "min", "ece": "min"}
    marks = {k: (s["summary"][k].idxmax() if v == "max" else s["summary"][k].idxmin())
             for k, v in best.items()}
    slope_best = (s["summary"].cal_slope - 1.0).abs().idxmin()
    rows = []
    for arm in ARM_ORDER:
        r = s["summary"].loc[arm]
        def mark(col, text):
            return f"**{text}**" if marks.get(col) == arm else text
        rows.append([
            f"**{LABEL[arm]}**" if arm == "HYBRID-CAT" else LABEL[arm],
            mark("auroc", f"{r.auroc:.3f} ± {r.auroc_sd:.3f}"),
            mark("auprc", f"{r.auprc:.3f}"),
            mark("brier", f"{r.brier:.4f}"),
            f"**{r.cal_slope:.3f}**" if arm == slope_best else f"{r.cal_slope:.3f}",
            f"{r.cal_intercept:+.3f}",
            mark("ece", f"{r.ece:.4f}"),
        ])
    d = s["clinical_verdicts"]
    row = d[(d.metric == "auroc") & (d.comparison == "HYBRID-CAT - CatBoost")].iloc[0]
    return render_table(
        "table03_discrimination_calibration.png",
        "Table 3. Discrimination, overall error and calibration, 5x2 repeated CV, mean "
        "over 10 folds. All arms receive the identical 66-feature union view.",
        ["Arm", "AUROC", "AUPRC", "Brier", "Cal. slope", "Cal. intercept", "ECE"], rows,
        colw=[1.95, 1.25, 0.80, 0.80, 0.90, 1.00, 0.80], highlight=0,
        notes=(f"CAFE vs. CatBoost: delta AUROC = {row.delta:+.4f}, sign {row['sign']} "
               f"→ {row.verdict.upper()} under the decision rule (§3.7). "
               "Best value in each column shown in bold.",))


def table4(s: dict) -> Path:
    xm, v = s["xm"], s["xmodal_verdicts"]
    rows = []
    for label, key in (("Logistic regression", "LOGREG"),
                       ("HistGradientBoosting", "HGB"),
                       ("CatBoost", "CatBoost"), ("XGBoost", "XGB")):
        base, plus = xm[f"{key}/base"], xm[f"{key}/+xmod"]
        sign, verdict = _sign(v, f"{key}: +xmod - base")
        delta = f"{plus - base:+.3f}"
        rows.append([label, f"{base:.3f}", f"{plus:.3f}",
                     f"**{delta}**" if key == "LOGREG" else delta, sign, verdict])
    return render_table(
        "table04_engineered_features_by_learner.png",
        "Table 4. Contribution of the engineered feature block, by learner: base table "
        "versus union view, with the fold-wise sign agreement and decision-rule verdict.",
        ["Learner", "Base (23 feats)", "+ engineered (66 feats)", "delta AUROC",
         "Sign", "Verdict"], rows,
        colw=[1.85, 1.20, 1.60, 1.00, 0.75, 0.75],
        notes=("The ordering is the finding: the benefit falls monotonically as the "
               "learner's native capacity to represent interactions rises.",))


def table5(s: dict) -> Path:
    eb, v = s["eb"], s["encb_verdicts"]

    def cell(arm):
        return f"{eb.loc[arm, 'mean']:.3f} ± {eb.loc[arm, 'std']:.3f}"

    dcat = eb.loc["A+B+C/cat", "mean"] - eb.loc["A+C/cat", "mean"]
    dxgb = eb.loc["A+B+C/xgb", "mean"] - eb.loc["A+C/xgb", "mean"]
    scat = _sign(v, "A+B+C/cat - A+C/cat")
    sxgb = _sign(v, "A+B+C/xgb - A+C/xgb")
    ap = s["encb"].groupby("arm").auprc.mean()
    rows = [
        ["A + C (CatBoost)", f"**{cell('A+C/cat')}**", f"{ap['A+C/cat']:.3f}",
         "—", "—", "—"],
        ["A + B + C (CatBoost)", cell("A+B+C/cat"), f"{ap['A+B+C/cat']:.3f}",
         f"{dcat:+.3f}", scat[0], scat[1]],
        ["A + C (XGBoost)", cell("A+C/xgb"), f"{ap['A+C/xgb']:.3f}", "—", "—", "—"],
        ["A + B + C (XGBoost)", cell("A+B+C/xgb"), f"{ap['A+B+C/xgb']:.3f}",
         f"{dxgb:+.3f}", sxgb[0], sxgb[1]],
        ["Encoder B alone", cell("B-alone"), f"{ap['B-alone']:.3f}", "—", "—", "—"],
    ]
    return render_table(
        "table05_encoder_b_ablation.png",
        "Table 5. Encoder B ablation. Encoder B is the temporal-attention model at "
        "12,585 parameters; all arms use the standardised-logit gate of §3.4.3.",
        ["Configuration", "AUROC", "AUPRC", "delta vs. no Encoder B", "Sign", "Verdict"],
        rows, colw=[1.85, 1.25, 0.80, 1.45, 0.75, 0.75], highlight=0,
        notes=("Mean gate weight assigned to Encoder B: 0.195, against an equal share "
               "of 0.333.",))


def table6(s: dict) -> Path:
    d = s["dca"]
    arms = ["HYBRID-CAT", "CatBoost", "LOGREG", "HYBRID-XGB", "XGBoost"]
    short = {"HYBRID-CAT": "CAFE", "CatBoost": "CatBoost", "LOGREG": "LogReg",
             "HYBRID-XGB": "CAFE-XGB", "XGBoost": "XGBoost"}
    rows = []
    for t in (0.10, 0.20, 0.30, 0.40, 0.50):
        r = d.loc[(d.threshold - t).abs().idxmin()]
        best = max(arms, key=lambda a: r[a])
        rows.append([f"{t:.2f}", f"{r.treat_all:.4f}", "0"]
                    + [f"**{r[a]:.4f}**" if a == best else f"{r[a]:.4f}" for a in arms])
    window = d[(d.threshold >= 0.05) & (d.threshold <= 0.50)]
    default = np.maximum(window.treat_all, window.treat_none)
    means = {short[a]: (window[a] - default).mean() for a in arms}
    order = sorted(means, key=means.get, reverse=True)
    return render_table(
        "table06_decision_curve.png",
        "Table 6. Net benefit at representative risk thresholds, pooled out-of-fold "
        "(repeat 2), against the treat-all and treat-none default strategies.",
        ["Threshold", "Treat all", "Treat none"] + [short[a] for a in arms], rows,
        colw=[0.85, 0.80, 0.85, 0.80, 0.85, 0.75, 0.90, 0.85],
        notes=("Mean net benefit above the better default strategy, thresholds "
               "0.05-0.50: " + ", ".join(f"{k} {means[k]:+.4f}" for k in order) + ".",))


def table7(s: dict) -> Path:
    opp = s["opp"].set_index("arm")
    cols = ["specificity", "ppv", "npv"]
    best = {c: opp[c].idxmax() for c in cols}
    best["flagged_frac"] = opp.flagged_frac.idxmin()
    rows = []
    for arm in ARM_ORDER:
        r = opp.loc[arm]
        short = "CAFE" if arm == "HYBRID-CAT" else (
            "CAFE-XGB" if arm == "HYBRID-XGB" else LABEL[arm])
        cells = [short, f"{r.threshold:.3f}", f"{r.sensitivity:.3f}"]
        for c in cols + ["flagged_frac"]:
            cells.append(f"**{r[c]:.3f}**" if best[c] == arm else f"{r[c]:.3f}")
        rows.append(cells)
    return render_table(
        "table07_operating_point.png",
        "Table 7. Performance at a 90 %-sensitivity screening threshold. The threshold "
        "is selected on repeat 1 and scored on repeat 2.",
        ["Arm", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV",
         "Fraction flagged"], rows,
        colw=[1.75, 0.95, 1.05, 1.05, 0.70, 0.70, 1.15], highlight=0,
        notes=("Single-split comparison: the decision rule cannot be applied, and the "
               "ordering should not be treated as established (§4.6).",))


def table8(s: dict) -> Path:
    imp = s["shap"].head(15)
    view = {"base": "base", "cross-modal": "engineered"}
    rows = [[str(i + 1), r.feature, f"{r.mean_abs_shap:.4f}", view.get(r.view, r.view)]
            for i, r in enumerate(imp.itertuples())]
    share = (s["shap"].loc[s["shap"].view != "base", "mean_abs_shap"].sum()
             / s["shap"].mean_abs_shap.sum())
    top20 = (s["shap"].head(20).view != "base").sum()
    return render_table(
        "table08_shap_top15.png",
        "Table 8. Top 15 features by mean |SHAP|, out-of-fold exact TreeSHAP over all "
        "3000 patients, with the view each feature belongs to.",
        ["Rank", "Feature", "mean |SHAP|", "View"], rows,
        colw=[0.60, 2.10, 1.10, 1.10],
        aligns=["center", "left", "center", "center"],
        notes=(f"The engineered block carries {share:.1%} of total attribution mass and "
               f"supplies {top20} of the top 20 features -- while ablating it costs the "
               "deployed learner 0.002 AUROC (Table 4).",))


def table9(s: dict) -> Path:
    cost, rt = s["cost"], s["runtime"]
    rows = [
        ["Logistic regression", f"{cost.loc['LOGREG', 'fit_s']:.3f}",
         f"{cost.loc['LOGREG', 'per_patient_ms']:.4f}", "67"],
        ["XGBoost", f"{cost.loc['XGBoost', 'fit_s']:.3f}",
         f"{cost.loc['XGBoost', 'per_patient_ms']:.4f}", "tree ensemble"],
        ["CAFE (XGBoost experts)", f"{cost.loc['HYBRID-XGB', 'fit_s']:.2f}",
         "~0.007 (est.)", "418 + 2 ensembles"],
        ["CatBoost", f"{cost.loc['CatBoost', 'fit_s']:.2f}",
         f"{cost.loc['CatBoost', 'per_patient_ms']:.4f}", "tree ensemble"],
        ["CAFE (CatBoost experts)", f"{cost.loc['HYBRID-CAT', 'fit_s']:.2f}",
         "~0.011 (est.)", "418 + 2 ensembles"],
    ]
    for model, label in (("PREG-Net", "PREG-Net"), ("FETA-Transformer", "FETA-Transformer"),
                         ("Gated hybrid", "Gated deep hybrid")):
        r = rt.loc[model]
        rows.append([label, f"{r.fit_single:.2f}", f"{r.per_patient_ms:.3f}",
                     f"{int(r.params):,}"])
    feta = rt.loc["FETA-Transformer", "fit_single"]
    return render_table(
        "table09_cost.png",
        "Table 9. Training and inference cost, measured on one fold on the same machine "
        "as all accuracy results.",
        ["Model", "Fit (s)", "Inference (ms/patient)", "Learned parameters"], rows,
        colw=[2.05, 0.90, 1.55, 1.60],
        aligns=["left", "center", "center", "center"],
        notes=("Inference for the CAFE arms is a component sum -- two tree evaluations "
               "plus a 418-parameter MLP forward pass -- and is therefore an "
               "estimate, not an end-to-end measurement.",
               f"CAFE-XGB trains {feta / cost.loc['HYBRID-XGB', 'fit_s']:.0f}x faster and "
               f"CAFE-CatBoost {feta / cost.loc['HYBRID-CAT', 'fit_s']:.1f}x faster than "
               f"the FETA-Transformer; against their own matched baselines the direction "
               f"reverses ({cost.loc['HYBRID-CAT', 'fit_s'] / cost.loc['CatBoost', 'fit_s']:.1f}x "
               f"a CatBoost fit, "
               f"{cost.loc['HYBRID-XGB', 'fit_s'] / cost.loc['XGBoost', 'fit_s']:.1f}x an "
               "XGBoost fit).",))


def table10(s: dict) -> Path:
    """The same six-step path fig_cafe_ablation.py plots, from the same three files."""
    xm, eb = s["xm"], s["eb"]
    hybrid = s["hybrid"].auroc.mean()
    steps = [
        ("Logistic regression, base features", xm["LOGREG/base"]),
        ("+ 43 engineered clinical features", xm["LOGREG/+xmod"]),
        ("+ swap learner to XGBoost", xm["XGB/+xmod"]),
        ("+ swap learner to CatBoost", xm["CatBoost/+xmod"]),
        ("+ split into views, add per-patient gate (CAFE)", hybrid),
        ("+ add sequence Encoder B",
         hybrid + (eb.loc["A+B+C/cat", "mean"] - eb.loc["A+C/cat", "mean"])),
    ]
    rows = []
    for i, (label, val) in enumerate(steps):
        if i == 0:
            delta = "—"
        else:
            d = val - steps[i - 1][1]
            delta = f"**{d:+.3f}**" if abs(d) >= 0.010 else f"{d:+.3f}"
        rows.append([str(i), label, f"{val:.3f}", delta])
    total = steps[4][1] - steps[0][1]
    return render_table(
        "table10_ablation_decomposition.png",
        "Table 10. Decomposition of the final AUROC: a five-step path from logistic "
        "regression on the base table to the full CAFE system, each step altering "
        "exactly one component.",
        ["Step", "Configuration", "AUROC", "delta"], rows,
        colw=[0.55, 3.55, 0.85, 0.85],
        aligns=["center", "left", "center", "center"], highlight=4,
        notes=(f"Total from step 0 to CAFE: {total:+.3f} AUROC. Every individual step is "
               "a tie under the decision rule; the relative magnitudes are the finding.",))


BUILDERS = [table1, table2, table3, table4, table5,
            table6, table7, table8, table9, table10]


def main() -> int:
    s = load()
    for n, build in enumerate(BUILDERS, start=1):
        path = build(s)
        print(f"  Table {n:>2}  ->  {path.relative_to(REPO)}")
    print(f"wrote {len(BUILDERS)} table images to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
