# CAFE — Tables and Figures for Insertion

Companion to `CAFE_Hybrid_Manuscript.md`. Every table below is numbered to match a
`[TABLE n HERE]` placeholder in the manuscript, and every figure entry matches a
`[FIGURE n HERE]` placeholder. Insert in order; captions are reproduced here so the caption
travels with the artifact.

All values are checked against their source CSVs by
`results/xmodal_study_2026-07-22/verify_cafe_manuscript.py`, which fails on any drift.

---

## Table 1 — The five feature families

*Caption:* **Table 1.** The five feature families: the clinical question each encodes, its
construction, and the number of features contributed.

| Family | Clinical question it encodes | Construction | n |
|---|---|---|---|
| Gestational normalisation | Is this value normal *for this week*? | z-score of each measurement within its rounded GA-week bin | basis |
| Temporal trajectory | Where is the marker now, on average, and at its worst? | per-patient last / mean / min of each z-trajectory | 12 |
| Growth velocity | Is the marker moving in the right direction, fast enough? | per-patient OLS slope of each z-trajectory against GA | 4 |
| Discordance | Do the markers *disagree with each other*? | within-patient correlation, level gap and slope gap for (FHR,CRL), (CRL,GS), (FHR,GS), (CRL,YSD); slope-sign disagreement; spread and worst channel across markers | 15 |
| Maternal risk profile | What is the patient's baseline clinical risk? | standardised composite of age, BMI, previous loss, IVF conception, non-singleton | 1 |
| Cross-modal interaction | Does maternal risk *modify* the sonographic signal? | 11 products of maternal terms with sonographic terms (e.g. age × zFHR_last, BMI × zCRL_last, composite × discordance spread) | 11 |

*Total added: 43 features, over a 23-feature base table, giving a 66-feature union view.*

---

## Table 2 — Encoder specifications

*Caption:* **Table 2.** Encoder specifications: the view each encoder consumes, its model
class, and its parameter count.

| Encoder | View | Model | Parameters |
|---|---|---|---|
| A | Flat tabular base table (23 features) | Gradient-boosted tree (XGBoost or CatBoost) | tree ensemble |
| B | Raw scan sequence (2–5 visits × 4 measures + GA) | Temporal attention, `d_model` 24, 1 layer, 2 heads | 12,585 |
| C | Cross-modal engineered block (43 features) | Gradient-boosted tree (XGBoost or CatBoost) | tree ensemble |
| Gate | Base table (23 features) | MLP 23 → 16 → *K*, softmax over experts | 418 (*K* = 2) |

---

## Table 3 — Discrimination, overall error and calibration

*Caption:* **Table 3.** Discrimination, overall error and calibration, 5×2 repeated CV, mean
over 10 folds. All arms receive the identical 66-feature union view. CAFE denotes the gated
two-expert configuration (Encoders A and C); the suffix indicates the boosting component
from which both experts are built. Best value in each column shown in bold.

| Arm | AUROC | AUPRC | Brier | Cal. slope | Cal. intercept | ECE |
|---|---|---|---|---|---|---|
| **CAFE (CatBoost experts)** | **0.852 ± 0.022** | 0.686 | 0.1048 | **0.946** | +0.039 | **0.0296** |
| CatBoost | 0.851 ± 0.023 | **0.692** | **0.1042** | 0.923 | +0.095 | 0.0370 |
| Logistic regression | 0.837 ± 0.019 | 0.643 | 0.1122 | 0.912 | −0.093 | 0.0376 |
| CAFE (XGBoost experts) | 0.833 ± 0.022 | 0.659 | 0.1165 | 0.480 | −0.148 | 0.0820 |
| XGBoost | 0.830 ± 0.027 | 0.653 | 0.1197 | 0.430 | −0.170 | 0.0932 |

*CAFE vs. CatBoost: Δ AUROC = +0.0008, sign 6/10 → **TIE** under the decision rule (§3.7).*

*Source:* `clinical_per_fold.csv` → `clinical_summary.csv`

---

## Table 4 — Contribution of the engineered feature block, by learner

*Caption:* **Table 4.** Contribution of the engineered feature block, by learner: base table
versus union view, with the fold-wise sign agreement and decision-rule verdict.

| Learner | Base (23 feats) | + engineered (66 feats) | Δ AUROC | Sign | Verdict |
|---|---|---|---|---|---|
| Logistic regression | 0.822 | 0.837 | **+0.015** | 10/10 | tie |
| HistGradientBoosting | 0.832 | 0.837 | +0.005 | 7/10 | tie |
| CatBoost | 0.850 | 0.851 | +0.002 | 7/10 | tie |
| XGBoost | 0.830 | 0.830 | +0.000 | 4/10 | tie |

*The ordering is the finding: benefit is monotonically inversely proportional to the
learner's native capacity to represent interactions.*

*Source:* `xmodal_per_fold.csv`, `xmodal_verdicts.csv`

---

## Table 5 — Encoder B ablation

*Caption:* **Table 5.** Encoder B ablation. Encoder B is the temporal-attention model at
12,585 parameters; all arms use the standardised-logit gate of §3.4.3.

| Configuration | AUROC | AUPRC | Δ vs. no Encoder B | Sign | Verdict |
|---|---|---|---|---|---|
| A + C (CatBoost) | **0.850 ± 0.021** | 0.684 | — | — | — |
| A + B + C (CatBoost) | 0.845 ± 0.022 | 0.680 | −0.005 | 8/10 | tie |
| A + C (XGBoost) | 0.832 ± 0.024 | 0.662 | — | — | — |
| A + B + C (XGBoost) | 0.829 ± 0.022 | 0.660 | −0.003 | 6/10 | tie |
| Encoder B alone | 0.813 ± 0.027 | 0.616 | — | — | — |

*Mean gate weight assigned to Encoder B: 0.195, against an equal share of 0.333.*

*Source:* `encoder_b_per_fold.csv`, `encoder_b_verdicts.csv`

---

## Table 6 — Net benefit at representative risk thresholds

*Caption:* **Table 6.** Net benefit at representative risk thresholds, pooled out-of-fold
(repeat 2), against the treat-all and treat-none default strategies.

| Threshold | Treat all | Treat none | CAFE | CatBoost | LogReg | CAFE-XGB | XGBoost |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.1148 | 0 | **0.1431** | 0.1412 | 0.1396 | 0.1259 | 0.1254 |
| 0.20 | 0.0042 | 0 | **0.1138** | 0.1138 | 0.1042 | 0.1020 | 0.1027 |
| 0.30 | −0.1381 | 0 | **0.0960** | 0.0940 | 0.0826 | 0.0863 | 0.0844 |
| 0.40 | −0.3278 | 0 | **0.0794** | 0.0783 | 0.0693 | 0.0699 | 0.0697 |
| 0.50 | −0.5933 | 0 | 0.0670 | **0.0687** | 0.0533 | 0.0610 | 0.0570 |

*Mean net benefit above the better default strategy, thresholds 0.05–0.50: CAFE **+0.0744**,
CatBoost +0.0737, LogReg +0.0648, CAFE-XGB +0.0626, XGBoost +0.0617. Spread between best and
worst arm 0.0128; gap from the worst arm to the defaults 0.0617.*

*Source:* `clinical_decision_curve.csv`

---

## Table 7 — Performance at a 90 %-sensitivity screening threshold

*Caption:* **Table 7.** Performance at a 90 %-sensitivity screening threshold. The threshold
is selected on repeat 1 and scored on repeat 2, so it is never chosen and reported on the
same predictions.

| Arm | Threshold | Sensitivity | Specificity | PPV | NPV | Fraction flagged |
|---|---|---|---|---|---|---|
| CAFE | 0.074 | 0.898 | **0.553** | **0.339** | **0.955** | **0.539** |
| CatBoost | 0.060 | 0.900 | 0.523 | 0.325 | 0.953 | 0.563 |
| Logistic regression | 0.081 | 0.903 | 0.497 | 0.314 | 0.953 | 0.584 |
| CAFE-XGB | 0.008 | 0.902 | 0.469 | 0.303 | 0.949 | 0.606 |
| XGBoost | 0.004 | 0.907 | 0.449 | 0.296 | 0.950 | 0.623 |

*Single-split comparison: the decision rule cannot be applied, and the ordering should not be
treated as established (§4.6).*

*Source:* `clinical_operating_point.csv`

---

## Table 8 — Top 15 features by mean |SHAP|

*Caption:* **Table 8.** Top 15 features by mean |SHAP|, out-of-fold exact TreeSHAP over all
3000 patients, with the view each feature belongs to.

| Rank | Feature | mean \|SHAP\| | View |
|---|---|---|---|
| 1 | `age` | 0.3092 | base |
| 2 | `mat_composite` | 0.3015 | engineered |
| 3 | `disc_last_CRL_YSD` | 0.2806 | engineered |
| 4 | `GS_slope` | 0.1841 | base |
| 5 | `zGS_last` | 0.1712 | engineered |
| 6 | `comp_x_disc_spread` | 0.1711 | engineered |
| 7 | `zFHR_slope` | 0.1701 | engineered |
| 8 | `FHR_latest` | 0.1373 | base |
| 9 | `zFHR_last` | 0.1361 | engineered |
| 10 | `bmi` | 0.1036 | base |
| 11 | `conception_ivf` | 0.0999 | base |
| 12 | `zYSD_mean` | 0.0956 | engineered |
| 13 | `CRL_slope` | 0.0940 | base |
| 14 | `zYSD_min` | 0.0938 | engineered |
| 15 | `zFHR_mean` | 0.0925 | engineered |

*The engineered block carries 66.8 % of total attribution mass and supplies 13 of the top 20
features — while ablating it costs the deployed learner 0.002 AUROC (Table 4).*

*Source:* `shap_global_importance.csv`

---

## Table 9 — Training and inference cost

*Caption:* **Table 9.** Training and inference cost, measured on one fold on the same machine
as all accuracy results. Deep-model figures are from the same programme's runtime benchmark
on the same cohort.

| Model | Fit (s) | Inference (ms/patient) | Learned parameters |
|---|---|---|---|
| Logistic regression | 0.040 | 0.0017 | 67 |
| XGBoost | 0.337 | 0.0033 | tree ensemble |
| CAFE (XGBoost experts) | 3.58 | ~0.007 † | 418 + 2 ensembles |
| CatBoost | 8.06 | 0.0050 | tree ensemble |
| CAFE (CatBoost experts) | 43.09 | ~0.011 † | 418 + 2 ensembles |
| PREG-Net | 266.32 | 1.071 | 16,130 |
| FETA-Transformer | 268.98 | 1.050 | 88,193 |
| Gated deep hybrid | 433.29 | 1.176 | 154,757 |

† Inference for the CAFE arms is a component sum — two tree evaluations plus a 418-parameter
MLP forward pass — rather than a direct end-to-end measurement, and is marked as an estimate.

*CAFE-XGB trains 75× faster and CAFE-CatBoost 6.2× faster than the FETA-Transformer. Against
their own matched baselines the direction reverses: 5.3× a single CatBoost fit and 10.6× a
single XGBoost fit.*

*Source:* `clinical_cost.csv`, `results/regime_study_2026-07-19/runtime_real_models.csv`

---

## Table 10 — Decomposition of the final AUROC

*Caption:* **Table 10.** Decomposition of the final AUROC: a five-step path from logistic
regression on the base table to the full CAFE system, each step altering exactly one
component.

| Step | Configuration | AUROC | Δ |
|---|---|---|---|
| 0 | Logistic regression, base features | 0.822 | — |
| 1 | + 43 engineered clinical features | 0.837 | **+0.015** |
| 2 | + swap learner to XGBoost | 0.830 | −0.007 |
| 3 | + swap learner to CatBoost | 0.851 | **+0.021** |
| 4 | + split into views, add per-patient gate (**CAFE**) | 0.852 | **+0.001** |
| 5 | + add sequence Encoder B | 0.847 | −0.005 |

*Total from step 0 to CAFE: +0.030 AUROC. Every individual step is a tie under the decision
rule; the relative magnitudes are the finding.*

*Source:* `xmodal_per_fold.csv`, `hybrid_catboost_experts_per_fold.csv`,
`encoder_b_per_fold.csv`

---

## Figures

All figures are pre-rendered PNGs. Regenerate with
`.venv/Scripts/python.exe results/xmodal_study_2026-07-22/make_cafe_figures.py`.

| # | File | Caption |
|---|---|---|
| 1 | `figures/cafe/fig_cafe_calibration.png` | **Figure 1.** Reliability curves, pooled out-of-fold. The two XGBoost-derived arms bow away from the diagonal in the same direction; the CatBoost-derived arms track it. |
| 2 | `figures/cafe/fig_cafe_decision_curve.png` | **Figure 2.** Net benefit against risk threshold, with treat-all and treat-none references. The five model curves are visually superimposed; all sit far above both defaults beyond a 15 % threshold. |
| 3 | `figures/cafe/fig_cafe_shap.png` | **Figure 3.** Top 20 features by mean \|SHAP\|, coloured by view. Engineered features occupy 13 of the top 20 places. |
| 4 | `figures/cafe/fig_cafe_accuracy_vs_cost.png` | **Figure 4.** AUROC against training time on a logarithmic axis. The frontier is flat: two orders of magnitude of additional compute buy approximately 0.02 AUROC, and the last 5× buys 0.001. |
| 5 | `figures/cafe/fig_cafe_ablation.png` | **Figure 5.** Decomposition of the final AUROC rendered as a waterfall: the architecture step is +0.001. |
