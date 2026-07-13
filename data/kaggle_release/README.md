# Synthetic Longitudinal First-Trimester Ultrasound Cohort (Pregnancy Loss Prediction)

**3,000 simulated pregnancies · 9,100 serial ultrasound scans · 20.6% first-trimester loss**

---

## ⚠️ Read this first: what this data is, and what it is not

**This is synthetic data. It contains no real patients and no real measurements.**
No individual in this dataset exists. It must **not** be used for clinical decisions,
and results obtained on it are **not** clinical evidence.

**But it is also not random numbers.** Every trajectory is generated from
*published* first-trimester growth curves and *published* pregnancy-loss effect
sizes, taken from peer-reviewed cohort studies (sources below). The generator
reproduces the real week-by-week reference medians for all four ultrasound
markers to within roughly 1–2 mm / a few bpm, and applies the loss-associated
abnormalities in the directions and magnitudes reported in the literature —
including the clinically important and often-missed detail that the **yolk sac is
*enlarged*, not shrunken, in pregnancies that go on to fail.**

In short: **simulated patients, real biology.** The intended purpose is
benchmarking and methods development for longitudinal / multimodal models when
patient-level first-trimester data is not publicly available — which, as far as I
could find, it currently is not.

---

## What's in the data

### `patients.csv` — one row per patient (3,000 rows)

| Column | Description |
| --- | --- |
| `patient_id` | Unique ID (`P00001` …) |
| `age` | Maternal age (years) |
| `bmi` | Maternal BMI (kg/m²) |
| `parity` | Prior births |
| `gravidity` | Total prior pregnancies (always ≥ parity + previous_loss) |
| `previous_loss` | 1 if prior pregnancy loss |
| `conception` | `Natural` or `IVF` |
| `singleton` | Always `True` (singletons only) |
| `label` | **Target.** 1 = first-trimester pregnancy loss, 0 = viable |

### `scans.csv` — one row per ultrasound scan (9,100 rows; 2–5 per patient)

| Column | Description |
| --- | --- |
| `patient_id` | Links to `patients.csv` |
| `gestational_age_weeks` | Gestational age at scan (~5–14 weeks) |
| `FHR` | Fetal heart rate (bpm) |
| `CRL` | Crown–rump length (mm) |
| `GS` | Gestational sac mean diameter (mm) |
| `YSD` | Yolk sac diameter (mm) |

Scans are **longitudinal and irregularly spaced** (mean 3.03 per patient, ~2 weeks
apart), which is the point: the data is designed for temporal models, not just
tabular ones.

---

## How it was built (the actual calibration)

**1. Viable growth curves = published reference medians.** The generator
interpolates the published week-by-week medians and extrapolates using the
reported growth rates (CRL +7.54 mm/wk, GS +6.65 mm/wk, YSD +0.38 mm/wk,
FHR +13.76 bpm/wk through week 10):

| Gest. week | CRL (mm) | GS (mm) | YSD (mm) | FHR (bpm) |
| --- | --- | --- | --- | --- |
| 6 | 5.0 | 16 | 2.6 | 116 |
| 7 | 11.5 | 24 | 2.9 | 148 |
| 8 | 18.3 | 31 | 3.3 | 168 |
| 9 | 25.4 | 37 | 3.7 | 171 |
| 10 | 35.8 | 43 | 4.1 | 170 |

The generated viable cohort reproduces these (e.g. observed week-8 CRL median
18.4 mm vs. reference 18.3 mm; week-10 GS 41.5 mm vs. 43 mm).

**2. Loss effects follow published odds ratios**, in the reported directions:

| Marker | Direction in loss | Published effect |
| --- | --- | --- |
| CRL | **smaller** for dates | OR 0.59 per mm (week 8) |
| GS | **smaller** | OR 0.01–0.10 |
| FHR | **slower** (bradycardia) | OR 0.96 per bpm (week 7) |
| YSD | **ENLARGED** | OR 3.36–6.52 (weeks 6–8); >6 mm predicts loss |

**3. Loss is heterogeneous, not uniform.** Real early loss has different
etiologies that present differently — so each simulated loss expresses only a
*subset* of these abnormalities, and about **18% are "occult"**: sonographically
indistinguishable from a viable pregnancy (as early genetic losses often are).
This is deliberate. It creates irreducible error and stops the task from being
trivially solvable.

**4. Realistic nuisance structure:** a constitutional size factor (some healthy
pregnancies are simply small), autoregressive scan-to-scan dependence,
marker-specific measurement noise, and clinically plausible missingness
(e.g., FHR often unmeasurable before 6.5 weeks) followed by interpolation.

**5. Weak maternal signal**, matching cohort evidence: loss vs viable differ only
slightly in age (~33.5 vs 31.5 yrs) and BMI (~26 vs 25), with prior loss carrying
most of the maternal signal.

---

## Benchmark difficulty (so you know what to expect)

5-fold cross-validated AUROC on aggregate per-patient features:

| Model / feature set | AUROC |
| --- | --- |
| Logistic regression (all aggregate features) | **0.887** |
| HistGradientBoosting (all aggregate features) | **0.880** |
| Ultrasound features only | 0.869 |
| Maternal features only | 0.580 |
| Single marker — FHR | 0.679 |
| Single marker — CRL | 0.652 |
| Single marker — YSD | 0.646 |
| Single marker — GS | 0.643 |

This sits in the same range as the published clinical models (AUC 0.86–0.92),
which is intentional. **The dataset is deliberately not saturated** — there is
real headroom for temporal and multimodal architectures to improve on tabular
baselines, and a hard ceiling imposed by the occult-loss fraction.

---

## Suggested uses

- Longitudinal / irregular time-series modeling (Transformers, RNNs, neural ODEs)
- Multimodal fusion (serial imaging markers + static maternal features)
- Graph-based modeling of inter-variable physiological relationships
- Class-imbalance methods (~1:4 positive:negative)
- Missing-data and irregular-sampling methods
- Teaching / reproducible benchmarking

## Limitations (please carry these into any write-up)

- **Synthetic.** No real patients. Not clinical evidence. Not for clinical use.
- Generated from *population-level* published statistics; it cannot capture
  unmodeled real-world structure (scanner/operator effects, comorbidities,
  site variation, ethnic variation in growth curves, twins, ectopics).
- The loss mechanism is a modeling assumption informed by literature, not a
  ground-truth causal model.
- Because the data is simulated, model performance here indicates **relative**
  methodological merit, not real-world accuracy.
- Any model trained on this **requires validation on real clinical data** before
  any claim of clinical utility.

## Reproducibility

Generated by `synthetic_generator_v2.py`, **seed 42**, `--n-patients 3000`,
`--loss-rate 0.20`. Fully deterministic — rerunning reproduces this file exactly.

## Sources the calibration is drawn from

1. *Early pregnancy ultrasound measurements and prediction of first trimester
   pregnancy loss: A logistic model.* **Scientific Reports** (2020). PMC6994659.
   — serial GS/YS/CRL/FHR from 6–10 weeks; per-week logistic models (AUC
   0.86–0.92); loss odds ratios; ~20% loss rate.
2. *Reference intervals of gestational sac, yolk sac, embryonic length, embryonic
   heart rate at 6–10 weeks after IVF-ET.* **PubMed 32928137** — 30,416
   pregnancies; growth models used to sanity-check curve shapes.
3. Population cohort studies of miscarriage risk factors (advanced maternal age,
   BMI, prior loss) used to set the maternal feature distributions.

## License

Released under **CC BY 4.0** — free to use with attribution. As it contains no
real patient data, there are no privacy or consent restrictions.
