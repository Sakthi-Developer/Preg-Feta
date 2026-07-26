# CAFE: Cross-modal Attention-gated Feature Ensemble

A clinically-grounded hybrid for **first-trimester pregnancy-loss prediction**: engineered
cross-modal features, per-view expert encoders, and a per-patient learned gate. It reaches
**AUROC 0.852** and is well calibrated — and it **ties the single gradient-boosted tree it
is built from**.

The paper reports that honestly and decomposes where the number came from:

| Source | Δ AUROC |
|---|---|
| Boosting component (XGBoost → CatBoost) | **+0.021** |
| Engineered cross-modal features (LOGREG / CatBoost) | +0.015 / +0.002 |
| **The gated fusion architecture the system is named for** | **+0.001** |
| Light sequence encoder as a third expert | **−0.005** |

The system is worth having. The explanation for why it works is not the one its
architecture suggests.

**All data are synthetic.** No patients were involved. The causal channel strengths are
specified parameters, not measured effect sizes — every clinical reading of this work is
constrained by that.

## The paper

- **`CAFE_Hybrid_Manuscript.md`** — the manuscript (`.docx` review copy alongside;
  `CAFE_Hybrid_TemplateFormat.docx` is the camera-ready two-column build).
- **`CAFE_Tables_and_Figures.md`** — table and figure supplement.
- **`CAFE_Paper_Assets.ipynb`** — renders and inspects every table and figure.
- `PREREGISTRATION.md` — the hypothesis and decision rule, fixed before the evaluation
  cohort existed.

## Layout

```
models/                            feta_transformer, preg_net, hybrid_fusion, ensemble
utils/                             training.py (unified Trainer), metrics.py
data/                              dataset.py, graph_builder.py, gen_hard/ (the cohort)
figures/cafe/                      Figures 1–5 and tables/ (Tables 1–10)
scripts/                           feature table, CV plumbing, parity check, docx render
results/xmodal_study_2026-07-22/   the CAFE study — every script and every result CSV
results/regime_study_2026-07-19/   the fusion-family benchmark cited in §5.4
results/cv_hard/                   per-fold predictions behind the pre-registered table
synthetic_generator_tuned.py       the synthetic cohort generator
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# (on macOS/Linux: source .venv/bin/activate && pip install -r requirements.txt)
```

The environment lives in `.venv/` (git-ignored). On Windows call
`.venv/Scripts/python.exe` directly — the system `python` does not have torch. **Run
everything from the repo root.**

## Quick start

```bash
# 1. verify the model wiring reads both modalities
.venv/Scripts/python.exe scripts/check_input_parity.py

# 2. check every number in the manuscript against its source CSV (163 of them)
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/verify_cafe_manuscript.py

# 3. regenerate the paper's figures and tables
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_figures.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_tables.py

# 4. recompute the study from scratch (slow)
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/xmodal_features.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/hybrid_better_experts.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/clinical_evaluation.py
```

## The evaluation cohort

`data/gen_hard` — 3000 patients, 9144 scan rows, 20.3 % prevalence, 2–5 irregularly spaced
visits each. Losses are generated as the logical OR of an independent **sonographic**
channel and a **maternal** channel, so the two modalities carry genuinely complementary
information and fusion is testable. This is a deliberate in-silico modelling assumption,
**not** a property of real pregnancies. Difficulty is calibrated so architecture can be
measured at all — on easier cohorts every arm saturates at 0.97–0.99.

Reproduce it bit-for-bit:

```bash
.venv/Scripts/python.exe synthetic_generator_tuned.py --out data/gen_hard \
    --etiology dual --seed 42 --n-patients 3000 --channel-corr 0.25 \
    --sono-mat-coupling 0.5 --mat-noise-sd 3.0 --sono-effect-scale 0.35
```

`channel_labels.csv` is **diagnostics only** — it holds the per-channel labels whose OR is
the target, so using it as a feature leaks the answer.

See `CLAUDE.md` for the full methodology rules, established results, and the list of tested
dead ends.
