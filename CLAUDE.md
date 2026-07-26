# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A research codebase for **first-trimester pregnancy-loss prediction** on synthetic data.
It began as a test of whether modality-specialised neural fusion beats simple models given
the same inputs — **it does not** — and the repository is now trimmed to the one paper that
ships: **CAFE**.

**`CAFE_Hybrid_Manuscript.md` is THE paper.** It builds the clinically-grounded hybrid
(engineered cross-modal features + per-view expert encoders + a per-patient learned gate),
reports it honestly at **AUROC 0.852**, and decomposes where that number actually came
from. It ties the single gradient-boosted tree it is built from. That is the result.

Everything else in this repo exists to support, reproduce, or verify that paper.

- **CAFE** — `results/xmodal_study_2026-07-22/`. Engineered features → view experts → gate.
- **FETA-Transformer** (`models/feta_transformer.py`) — temporal attention over the scan
  sequence. Used by the parity check and the runtime benchmark; CAFE's Encoder-B ablation
  reimplements a light version of it inline.
- **PREG-Net** (`models/preg_net.py`) — hand-rolled graph attention network over clinical
  variables. No PyTorch Geometric — **do not introduce `torch_geometric`.**
- **HybridFusionPredictor** (`models/hybrid_fusion.py`) — cross-attention fusion with a
  per-patient gate (`--gated`).
- **EnsemblePredictor** (`models/ensemble.py`) — late fusion baseline.

All data are **synthetic**. There is no real patient data. Models are pure PyTorch.

`.venv/` holds the environment (git-ignored). On Windows use `.venv/Scripts/python.exe`;
the system `python` lacks torch. **Always run from the repo root** — scripts do
`sys.path.insert(0, ROOT)`.

---

## 🚫 THE RULES — violating any of these invalidates the result

**1. Input parity.** Every model gets **both** modalities (`patients.csv` + `scans.csv`).
Do **not** pass `--clinical-only`. Parity is a property of the **wiring**, not the flags —
a cross-attention bug once starved FETA of the maternal modality for weeks while every
results table claimed otherwise. **Verify by gradient:**
```bash
.venv/Scripts/python.exe scripts/check_input_parity.py   # exits 1 on violation
```
Run it after **any** model or fusion wiring change.

**2. Tree baselines must use `scripts/scaling_sweep.py::build_feature_table_fast`.**
Ad-hoc features understate the tree by ~0.019 AUROC and manufacture false wins. **This
error was made twice.** Reach it through `published_families_benchmark.load()`.

**3. `channel_labels.csv` is diagnostics-only.** It holds `label_sono`/`label_mat` and
`label == label_sono OR label_mat`, so feeding it as a **feature** leaks the answer
(instant 1.00). Never a model input, never an evaluation target. *One sanctioned
exception:* as an auxiliary **training** target, restricted to training-split patients,
asserted disjoint from every test split, evaluated on the plain `label`.

**4. Report AUROC/AUPRC, never bare accuracy.** At 20.3 % prevalence, "predict no loss for
everyone" scores **79.7 %**.

**5. Use repeated CV.** 5-fold is unstable at these effect sizes — one effect halved from
+0.012 to +0.003 under 5×2. Use repeated CV before believing anything under ~0.02.

**6. Never compare our AUROC to published AUROC from other cohorts.** Different patients,
different prevalence, different difficulty. Invalid, and reviewers reject it.

**7. Never weaken a baseline to make the hybrid win.** If you give CAFE a tuning budget,
every baseline gets the same budget. Make the *task* harder, never the opponents weaker.

**8. Name the winner correctly.** The model that ties/beats the fusion architectures is
**CatBoost / HistGradientBoosting** — an off-the-shelf tree. It is **NOT PREG-Net**, which
scored **0.808, the lowest arm** in the pre-registered table. Any sentence of the form "the
simple model wins" must name **the tree**.

**9. Ties are ties.** A difference counts only if it exceeds **2× pooled fold SD** *and*
agrees in sign on ≥80 % of folds. Report ties as ties regardless of the sign of the point
estimate — including where a tie is inconvenient.

⚠️ **"Pooled SD" means the WITHIN-ARM pooled SD** — `sqrt((var(a)+var(b))/2)`. The shipped
`verdict()` in `xmodal_features.py` uses `np.std(concatenate([a,b]))`, which absorbs the
between-arm gap into the threshold and so gets *harder to pass the further apart the arms
are*. Harmless for AUROC here (ratio 0.95–1.04, and **all 56 pairwise AUROC verdicts in the
study are identical under both**), but it hit 2.97 on calibration slope and returned "tie"
for two distributions that **do not overlap at all** (XGB 0.310–0.495 vs CatBoost
0.684–1.061). When |Δ| approaches or exceeds the fold SD, **compute both** and report both.
CAFE §4.4.1 discloses this defect; do not quietly "fix" it without updating that section.

---

## The paper: CAFE

`CAFE_Hybrid_Manuscript.md` + `CAFE_Tables_and_Figures.md`. Headline table, 5×2 repeated CV
on `data/gen_hard`, 66-feature union view
(`results/xmodal_study_2026-07-22/clinical_summary.csv`):

| Arm | AUROC | AUPRC | Brier | Cal. slope | ECE |
|---|---|---|---|---|---|
| **CAFE (CatBoost experts)** | **0.852 ± 0.022** | 0.686 | 0.1048 | **0.946** | 0.0296 |
| CatBoost | 0.851 ± 0.023 | 0.692 | 0.1042 | **0.923** | 0.0370 |
| LOGREG | 0.837 ± 0.019 | 0.643 | 0.1122 | 0.912 | 0.0376 |
| CAFE (XGBoost experts) | 0.833 ± 0.022 | 0.659 | 0.1165 | **0.480** | 0.0820 |
| XGBoost | 0.830 ± 0.027 | 0.653 | 0.1197 | **0.430** | 0.0932 |

**The decomposition is the paper's actual contribution.** CAFE's 0.852 breaks down as:

| Source | Δ AUROC |
|---|---|
| Boosting component (XGB → CatBoost) | **+0.021** |
| Engineered cross-modal features, LOGREG | +0.015 |
| Engineered cross-modal features, CatBoost | +0.002 |
| **The gated fusion architecture it is named for** | **+0.001** |
| Light sequence encoder as a 3rd expert | **−0.005** |

**Calibration separates what AUROC cannot.** XGBoost's predicted log-odds are ~2× too
extreme; CatBoost's are nearly right. AUROC ties (0.021 apart) and Brier ties — **only the
slope shows it**, and the per-fold distributions don't overlap. Consequence: the
90 %-sensitivity threshold ranges **0.004 → 0.081** across arms, so a hard-coded risk
threshold is coupled to one model's calibration.
⚠️ `utils/metrics.py` still has **no** Brier/calibration/net-benefit — use
`clinical_evaluation.py`.

**The gate is not collapsed.** Mean 0.557, SD 0.402, full 0–1 range, near-bimodal, and it
routes legibly (abnormal YSD → cross-modal expert, high maternal risk → flat expert).
Confident, coherent, and worth +0.001. Do not re-open "the gate collapsed" — it's refuted.

**Verify before touching any number:**
```bash
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/verify_cafe_manuscript.py
# checks 163 numbers in both CAFE .md files against the source CSVs; exits 1 on drift
```

---

## The backdrop (established, do not re-litigate)

CAFE §5.4 positions itself against a pre-registered fusion-architecture benchmark from the
same programme. **That paper is no longer in this repo** (removed 2026-07-26 — see the last
section), but its evidence is retained because §5.4 cites it. Numbers below are the ones
§5.4 relies on; the CSVs backing them are in `results/regime_study_2026-07-19/`.

**Eight architecturally distinct fusion families are statistically indistinguishable** on
`data/gen_hard`, 5×2 repeated CV (`published_families_benchmark.csv`):

| Family | AUROC | AUPRC |
|---|---|---|
| **LATE-AVG** (plain average, no learned fusion) | **0.840 ± 0.019** | 0.669 |
| GBM | 0.832 ± 0.020 | 0.644 |
| EARLY | 0.832 ± 0.023 | 0.640 |
| LATE-STACK | 0.832 ± 0.020 | 0.655 |
| GATED / MoE | 0.827 ± 0.022 | 0.646 |
| DEEP2GBM | 0.825 ± 0.020 | 0.636 |
| INTERMEDIATE | 0.825 ± 0.023 | 0.647 |
| LOGREG | 0.822 ± 0.023 | 0.620 |

Total spread **0.018** — smaller than any single arm's fold SD (0.019–0.023). `GATED` and
`MoE` are the **same estimator** here (identical to machine precision) and are reported as
**one arm** — hence eight, not nine.

**The pre-registered test (`PREREGISTRATION.md`, fixed before `gen_hard` existed): TIE.**
Reproduce it from the retained per-fold predictions:
```bash
.venv/Scripts/python.exe scripts/analyze_prereg.py --run-dir results/cv_hard \
    --patients-csv data/gen_hard/patients.csv --scans-csv data/gen_hard/scans.csv
```

| Model | AUROC | Per-fold |
|---|---|---|
| **HistGradientBoosting** | **0.836 ± 0.013** | 0.857 0.842 0.835 0.827 0.819 |
| Gated hybrid | 0.825 ± 0.013 | 0.831 0.826 0.840 0.827 0.801 |
| Logit-average ensemble | 0.822 ± 0.019 | 0.791 0.843 0.840 0.825 0.810 |
| FETA | 0.818 ± 0.012 | 0.837 0.804 0.815 0.825 0.808 |
| PREG-Net | 0.808 ± 0.018 | 0.778 0.828 0.817 0.818 0.799 |

gap −0.011, pooled SD 0.013, sign 3/5 (needs ≥4) → **TIE**, sign negative.

**Not a tree artefact.** Four boosting libraries agree within 0.020 (`tree_implementations.csv`):
CatBoost 0.850, LightGBM 0.835, HGB 0.832, XGBoost 0.830.

**Compute is the other separating axis** (`runtime_real_models.csv`). Every fusion family
costs 277–570× the training time of logistic regression and 1.3–2.6× CatBoost, without
beating either:

| Model | Fit (s) | Predict (ms) | × slower |
|---|---|---|---|
| LOGREG | 0.011 | 0.20 | 1× |
| LightGBM | 0.080 | 2.84 | 7× |
| CatBoost | 2.418 | 2.16 | 220× |
| LATE-AVG | 4.727 | 0.63 | 429× |
| DEEP2GBM | 6.157 | 7.61 | 559× |
| GATED | 6.281 | 0.74 | 570× |

**Say "not more accurate at 1–2 orders of magnitude more compute" — never "less accurate."**

### Mechanism (why everything ties)

1. **Feature saturation.** Scan-only AUROC plateaus at 0.732 (30 feats) / 0.720 (25) /
   0.728 (59) against a measured oracle sono ceiling of **0.820**. Doubling features does
   nothing.
2. **Towers self-specialise.** With *zero* channel supervision the scan tower recovers
   `label_sono` at **0.889**. ⚠️ This **refutes** the intuitive "OR labels damage each
   specialist" story — tested, false, do not put it in any paper.
3. **Auxiliary supervision regularises, it does not inform.** Real channel labels 0.842 vs
   **shuffled control 0.839** — the control reproduces the gain.

---

## The cohort

**`data/gen_hard` is the only cohort on disk, and it is the evaluation cohort.** Every
number in CAFE is computed on it. The other 16 cohorts were removed on 2026-07-26; if you
need one, regenerate it rather than guessing at it.

| Property | Value |
|---|---|
| Etiology | dual, difficulty-calibrated |
| N patients | 3000 |
| Scan rows | 9144 |
| Prevalence | 20.3 % |
| `channel_labels.csv` | yes (diagnostics only — rule 3) |
| Oracle ceilings | 0.820 sono / 0.756 maternal / 1.000 both |

**Why `gen_hard` and not the easier published cohorts:** on `generated_dual` everything
saturated at **0.97–0.99**, so architecture could not be measured. `gen_hard` drops the tree
to 0.836 — inside the pre-registered band [0.83, 0.87] — so the comparison can resolve. **The
null is reported where a positive result was possible.**

### Scan shape — measure it, don't assume it

`gen_hard` scans are **2–5 visits per patient (mean 3.05), irregularly spaced** (gaps 1–4
weeks, mean 2.00, SD 0.49), 4 measures per visit. The `flat` feature set is 25 columns
because max-visits is 5 (5 slots × 5 values incl. GA).

⚠️ Earlier versions of this file said "5 regularly-spaced visits." **That was wrong on both
counts.** It matters: the data are *already* irregular and ragged, and a tree still ties
every temporal architecture. **Sequence *length* is what makes the task tabularisable, not
regularity** — so any "change the data's SHAPE" experiment must go after *many more visits*,
not irregular timing.

### Generator

`synthetic_generator_tuned.py` made every dual cohort. **Use this for any difficulty work.**
Live knobs: `--mat-noise-sd`, `--sono-effect-scale`. Verified byte-identical to commit
`b7ac269d` at default knobs.

Reproduce `gen_hard` bit-for-bit:
```bash
.venv/Scripts/python.exe synthetic_generator_tuned.py --out data/gen_hard \
    --etiology dual --seed 42 --n-patients 3000 --channel-corr 0.25 \
    --sono-mat-coupling 0.5 --mat-noise-sd 3.0 --sono-effect-scale 0.35
```
**The knobs are FROZEN — do not retune them.** Verified: prevalence 20.3 %,
`label == label_sono OR label_mat` at 1.0000.

⚠️ `synthetic_generator_v2.py` (the epi-grounded rewrite) was **removed** — its separability
knobs were dead code (`--sono-mat-coupling` accepted and silently ignored), which made it a
trap. Recover from git history only if you need the epidemiology model, never for difficulty
tuning.

---

## ☠️ Dead ends — do NOT re-try

Each was tested with a fair baseline and a control. Each tied. The raw run directories were
removed on 2026-07-26; these verdicts stand and do not need re-running.

| Idea | Result |
|---|---|
| Gated per-patient routing, 13 stress regimes | context effect −0.008…+0.002 |
| …under missingness / scan corruption / modality dropout | tie (even given an oracle "bad scan" hint) |
| …under small-N (400/800/1500), prevalence 0.10/0.05 | tie |
| Noisy-OR head matched to the generator | +0.003 vs logit avg = tie |
| Richer scan features (30 → 59) | 0.732 → 0.728, no movement |
| Channel supervision (aux loss on label_sono/mat) | killed by shuffled control |
| Two-tower split at all | joint tree wins **13/13** by 0.028–0.076 |
| Optimising only the hybrid until it wins | **refused** — baseline sandbagging, see rule 7 |
| 43 cross-modal engineered features (GA z-scores, discordance, velocity, maternal×sono interactions) | tie for all 4 learners; gain is **inversely proportional to the learner's interaction capacity** — LOGREG +0.015 (10/10), HGB +0.005, CatBoost +0.002, XGB +0.000 (4/10). Re-expression, not new information |
| Gated 2-view hybrid on **CatBoost** experts (= CAFE, 0.852) | ties CatBoost 0.851. Component swap XGB→Cat **+0.021**; the gate itself **+0.001** |
| Light sequence encoder (FETA @ d_model 24, **12,585 params**) as a 3rd expert | −0.005 (cat) / −0.003 (xgb); gate gives it **0.195** of mass vs 0.333 equal share |
| Gate collapse as the explanation for gates not helping | **refuted** — near-bimodal, routes legibly, worth +0.001 |

**0.95 is not reachable on `gen_hard` by honest means.** The signal was deliberately
destroyed by the difficulty knobs. Anything at 0.95 here is a **leak**, most likely
`channel_labels.csv`.

### Two self-inflicted errors that manufactured fake wins

- **Weak tree baseline, twice** — ad-hoc features gave HGB 0.817/0.826 instead of
  0.832–0.836. Both times an apparent win vanished on correction. → **rule 2**.
- **Learner/context confound** — comparing HGB-with-context against LogReg-without-context
  faked a −0.067 "routing penalty" that was entirely the learner. **Read a context effect
  DOWN a column with the learner held fixed, never across learners.**

### The one untested direction with a real mechanism

Change the data's **shape**: many more visits per patient, so the trajectory no longer
flattens into a fixed-width table. Making the data *easier* does not help — on saturated
cohorts everything ties at 0.98.

---

## Repository layout

Everything below is on the CAFE reproduction chain or is provenance CAFE cites. There is
nothing spare left — if you delete something here, check what imports it first.

```
CAFE_Hybrid_Manuscript.md          ← THE PAPER (+ .docx review copy)
CAFE_Hybrid_TemplateFormat.docx      camera-ready, ProComm two-column template
CAFE_Tables_and_Figures.md           table/figure supplement (+ .docx)
CAFE_Paper_Assets.ipynb              renders + inspects every table and figure
Full-Paper-template.docx             the Word template md_to_docx_template.py fills
PREREGISTRATION.md                   hypothesis + decision rule, fixed before gen_hard
synthetic_generator_tuned.py         THE generator (see above)
Research.pdf                         source literature the programme started from

models/    feta_transformer, preg_net, hybrid_fusion, ensemble
utils/     training.py (unified Trainer), metrics.py
data/      dataset.py, graph_builder.py, gen_hard/
figures/cafe/                        CAFE Figures 1–5 + tables/ (Tables 1–10)

scripts/
  check_input_parity.py     ← RUN AFTER ANY WIRING CHANGE (rule 1)
  scaling_sweep.py          build_feature_table_fast — THE tree feature table (rule 2)
  cross_validate.py         imported by scaling_sweep (folds, loaders, device)
  train_baselines.py        imported by both (MATERNAL_FEATURES / US_FEATURES)
  analyze_prereg.py         applies PREREGISTRATION.md's rule to results/cv_hard
  smoke_test.py             forward-pass check for all four models
  md_to_docx.py             plain review .docx
  md_to_docx_template.py    camera-ready .docx into Full-Paper-template.docx

results/xmodal_study_2026-07-22/     ← THE CAFE STUDY
  xmodal_features.py           43 engineered cross-modal features, 4 learners
  hybrid_better_experts.py     the gated 2-view hybrid on CatBoost experts
  encoder_b_lightattention.py  the light sequence encoder as a 3rd expert
  clinical_evaluation.py       calibration / decision curve / operating point / SHAP
  make_cafe_figures.py         → figures/cafe/*.png
  make_cafe_tables.py          → figures/cafe/tables/*.png
  verify_cafe_manuscript.py    163 numbers vs source CSVs; exits 1 on drift
  *.csv, shap_values_oof.npy   the recorded results

results/regime_study_2026-07-19/     provenance for CAFE §5.4
  published_families_benchmark.py    IMPORTED BY THE CAFE SCRIPTS — do not move it
  tree_implementations.csv, runtime_real_models.csv, regime_screen.csv, ...

results/cv_hard/                     per-fold predictions behind the prereg table
                                     (torch checkpoints dropped 2026-07-26)
```

---

## How to run

```bash
# sanity
.venv/Scripts/python.exe scripts/smoke_test.py
.venv/Scripts/python.exe scripts/check_input_parity.py     # MUST pass before any run

# verify the paper, then regenerate its assets
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/verify_cafe_manuscript.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_figures.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_tables.py

# recompute the study itself (slow)
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/xmodal_features.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/hybrid_better_experts.py
.venv/Scripts/python.exe results/xmodal_study_2026-07-22/clinical_evaluation.py

# the pre-registered analysis (provenance for §5.4)
.venv/Scripts/python.exe scripts/analyze_prereg.py --run-dir results/cv_hard \
    --patients-csv data/gen_hard/patients.csv --scans-csv data/gen_hard/scans.csv

# render the manuscript
.venv/Scripts/python.exe scripts/md_to_docx_template.py CAFE_Hybrid_Manuscript.md
```

**GPU:** scripts default to `--device auto` (CUDA > MPS > CPU). `auto` silently falls back
to CPU when torch has no CUDA build — confirm with:
```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want 2.13.0+cu130 True   — a bare 2.13.0 or +cpu is the CPU-only wheel
```
The CUDA wheel comes from PyTorch's own index, and the series must match the driver
(cu130 needs ≥ 580).

---

## Conventions

- Class imbalance via `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`; models emit **raw
  logits**. Checkpoint/early-stop on **validation AUROC**.
- `utils/training.py::Trainer` trains all families; `collate_mode ∈ {feta, preg, ensemble}`
  selects the forward signature.
- Keep attention-dict keys stable — downstream code depends on them.
- The `.py` modules are the source of truth; `CAFE_Paper_Assets.ipynb` **imports** them, so
  a signature change breaks the notebook too.
- Long runs: this machine's CPU counters are unreliable — diagnose stalled jobs by **file
  mtime**. Background bash is capped at 10 min; for longer jobs launch detached via
  PowerShell `Start-Process` and watch for the output file.

## Reporting discipline (for any manuscript text)

- **Data are synthetic** — say so in every artifact. Channel strengths are **specified, not
  estimated**; the maternal channel is near-deterministic **by construction**.
- The cohort is **dual-etiology by design**, built so complementary signal exists for fusion
  to exploit. A null here is informative *because* the data favoured fusion.
- CAFE is a **good system with the wrong-sounding explanation**. Its performance lives in
  the boosting component, not the fusion architecture it is named for. Never let the
  architecture take credit for the +0.021 that belongs to CatBoost.
- We re-implemented published **architectural families**, not other authors' models or
  code. We took their **designs, not their results**. Never claim we beat their systems.

---

## What was removed on 2026-07-26

The repo was cut from 414 MB to 14 MB, on branch `clean/cafe-only`. **Do not go looking for
these — they are gone on purpose.**

- **All non-CAFE manuscripts**: the negative-result paper and its `.docx`/repro notebook,
  the gated-fusion and original manuscripts, the compute-cost short paper, the FullAnalysis
  notebook/HTML/PDF, `FETA_PREG*.ipynb`, `PROJECT_BRIEF.md`.
- **`figures/paper_assets/`, `figures/analysis/`, `figures/fig*.png`, `figures/negresult_*.png`**
  — every non-CAFE figure, plus `scripts/make_paper_assets.py` which built them.
- **16 of 17 data cohorts** and **all results run-dirs except `cv_hard`** and the two study
  directories.
- **`scripts/`**: `compare_models.py`, `train_{feta,preg,hybrid,ensemble}.py`,
  `overfit_batch.py`, `tune_sweep.py`, `inject_missingness.py`,
  `run_missingness_sweep.sh`, `make_paper_assets.py`.
- **`synthetic_generator_v2.py`** (dead separability knobs).

Files that were **tracked** in git are recoverable from history before the cleanup commit.
Files that were **untracked** — including the negative-result manuscripts,
`figures/paper_assets/`, and every deleted `results/` run-dir — were never committed and are
**permanently gone**. The verdicts they supported are transcribed above and in CAFE §5.4;
that transcription is now the only record.
