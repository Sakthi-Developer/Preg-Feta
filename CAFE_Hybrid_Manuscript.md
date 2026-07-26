# Clinically-Aligned Fusion of Experts for First-Trimester Pregnancy-Loss Prediction: An Interpretable, Compute-Efficient Hybrid and a Component-Level Ablation of Its Accuracy

**Mr. A. V. M. Kumaran¹\*, Dr. M. Sundara Rajan²**

¹Research Scholar, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035. Email: avmprofessor@gmail.com

²Associate Professor, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035.

*\*Corresponding author:* Mr. A. V. M. Kumaran. Email: avmprofessor@gmail.com

---

## Abstract

**Background.** Multimodal clinical prediction is conventionally approached by engineering
domain-specific features, assigning specialist encoders to different views of the patient,
and combining those encoders with a learned per-patient gate. The premise is that
modality-specific specialisation recovers signal a single joint model would miss. That
premise is widely adopted and rarely tested against its own counterfactual.

**Objective.** To build such a system for first-trimester pregnancy-loss prediction, to
evaluate it beyond discrimination alone, and to determine by controlled ablation which of
its components actually produce its accuracy.

**Methods.** We developed **CAFE (Clinically-Aligned Fusion of Experts)**: 43 engineered
features in five clinically motivated families — gestational normalisation, temporal
trajectory, growth velocity, inter-marker discordance, and maternal risk profile — together
with tree experts over a flat view and a cross-modal view, an optional 12,585-parameter
temporal attention encoder over the raw scan sequence, and a per-patient softmax gate over
expert logits. Evaluation used a synthetic dual-etiology cohort (n = 3000, prevalence
20.3 %) under 5×2 repeated stratified cross-validation. Every arm, hybrid and baseline
alike, received an identical 66-feature union view, so no result can be attributed to an
information advantage. Reported measures were AUROC, AUPRC, Brier score, Cox calibration
slope and intercept, expected calibration error, decision-curve analysis, and a screening
operating point selected on one cross-validation repeat and scored on the other. A
difference was declared real only if it exceeded twice the pooled fold standard deviation
and agreed in sign on at least 8 of 10 folds.

**Results.** CAFE reached AUROC 0.852 — the highest point estimate recorded in this research
programme and inside the 0.840–0.855 band predicted for it in advance — with a calibration
slope of 0.946 and expected calibration error of 0.030 without post-hoc correction, the
highest mean net benefit across risk thresholds 0.05–0.50, and a training time of 43 s
against 269 s for the deep temporal architecture it replaces. Its accuracy, however, does
not originate in its architecture. Replacing the gated multi-view ensemble with a single
gradient-boosted tree on the same features changed AUROC by +0.001. Decomposing the +0.030
total improvement over logistic regression attributed **+0.021 to the choice of boosting
library**, **+0.015 to the engineered features but only for a learner unable to represent
interactions natively** — the same features were worth +0.002 to the deployed learner, and
their benefit across four learners was monotonically inversely proportional to native
interaction capacity — **+0.001 to the gated fusion**, and **−0.005 to the sequence
encoder**, which the gate independently learned to discount (weight 0.195 against an equal
share of 0.333). Two further findings emerged from evaluating beyond discrimination. Feature
importance and feature contribution pointed in opposite directions: out-of-fold exact
TreeSHAP assigned the engineered block 66.8 % of total attribution mass and ranked a
discordance feature third overall, while ablation valued that block at 0.002 AUROC for the
deployed model. And the gate proved confidently per-patient and clinically legible yet
inconsequential: its weights were near-bimodal (mean 0.557, SD 0.402, spanning 0–1), routing
patients with abnormal yolk-sac measurements to the cross-modal expert and patients with
prominent maternal risk to the flat expert — a policy nobody specified, and worth +0.001.

**Conclusions.** The only dimension along which the arms genuinely separated was
calibration, which no part of the original proposal had identified as the place to look:
XGBoost-derived arms had calibration slopes of 0.430–0.480 against 0.923–0.946 for
CatBoost-derived arms, with non-overlapping per-fold distributions, while their AUROCs
differed by 0.021 and their Brier scores tied. Consequently the 90 %-sensitivity thresholds
spanned 0.004 to 0.081 across arms, so any workflow with a hard-coded risk threshold is
silently coupled to one model's calibration. CAFE is worth deploying; the explanation for
why it performs as it does is not the one its architecture suggests. All data are synthetic;
the findings concern modelling methodology and carry no clinical implication for real
pregnancies.

**Keywords:** multimodal fusion, mixture of experts, clinical feature engineering, model
calibration, decision-curve analysis, SHAP, ablation, early pregnancy loss, synthetic
clinical benchmarks

---

## Abbreviations

| Term | Expansion |
|---|---|
| AUROC | Area under the receiver-operating-characteristic curve |
| AUPRC | Area under the precision–recall curve |
| BMI | Body mass index |
| CAFE | Clinically-Aligned Fusion of Experts (the system proposed here) |
| CRL | Crown–rump length |
| CV | Cross-validation |
| DCA | Decision-curve analysis |
| ECE | Expected calibration error |
| FHR | Fetal heart rate |
| GA | Gestational age |
| GS | Mean gestational-sac diameter |
| HGB | HistGradientBoosting |
| IVF | In-vitro fertilisation |
| MLP | Multilayer perceptron |
| MoE | Mixture of experts |
| NPV | Negative predictive value |
| OOF | Out of fold |
| PPV | Positive predictive value |
| SHAP | SHapley Additive exPlanations |
| YSD | Yolk-sac diameter |

---

## 1. Introduction

### 1.1 The problem, and the shape of the data

Approximately one in five clinically recognised pregnancies ends in first-trimester loss.
The data available to a clinician at that stage are inherently multimodal and asymmetric: a
short, irregularly spaced sequence of ultrasound examinations, each yielding a handful of
biometric measurements, set against a fixed panel of maternal clinical covariates recorded
once at booking.

Both halves of that description are awkward for modern modelling. The sequence is far too
short for the sequence models that dominate longitudinal healthcare work — two to five
visits, not hundreds of time steps — and the covariate panel is too small for representation
learning to have much to discover. This is an unglamorous regime, and it is the regime that
most first-trimester prediction problems actually occupy.

### 1.2 The prevailing architectural response

The dominant modelling response has been architectural. A temporal encoder is assigned to
the scan trajectory, a separate encoder to the maternal covariates, and a learned fusion
mechanism — concatenation, cross-attention, or a gate — combines them. The implicit premise
is that modality-specific specialisation recovers signal that a single joint model would
miss.

The premise is plausible, and it is the reason such systems are built. It is also, in the
published literature, seldom accompanied by the measurement that would establish it: the
same information, given whole to one ordinary model.

### 1.3 What this paper delivers, in plain terms

We built the hybrid that this line of reasoning recommends, in the specific form a reviewer
of our earlier work proposed: five families of clinically-aligned engineered features, three
encoders operating on different *views* of the same patient — a flat tabular view, a
sequence view, and an interaction view — and a per-patient gate that learns how much to
trust each encoder for each individual patient.

**It works.** It reaches AUROC 0.852 on our evaluation cohort, the highest point estimate
recorded anywhere in this research programme. It trains in well under a minute per fold. It
is well calibrated with no post-hoc correction. Its predictions decompose into per-patient
feature attributions a clinician can read, and its gate emits a per-patient trust score for
each encoder.

**And its accuracy does not come from its architecture.** When we hold everything else fixed
and remove the fusion mechanism — replacing the gated ensemble of view-specific experts with
a single gradient-boosted tree trained on the union of the same features — the score moves
by +0.001 AUROC, far inside noise. The engineered features do contribute, but overwhelmingly
to *weak* learners: they lift logistic regression by +0.015 and CatBoost by +0.002. The
single largest contributor to the final number is neither the features nor the fusion but
the choice of boosting library underneath, worth +0.021 on its own. The sequence encoder,
given a fair budget and a fair gate, contributes −0.005.

We report this because the alternative — presenting 0.852 as evidence that gated multi-view
fusion is what produced it — would be a specific, checkable, and false causal claim, and
because a system whose active ingredient has been misidentified cannot subsequently be
improved. The engineering is sound and worth deploying. The explanation for why it performs
as it does needs to be the correct one.

### 1.4 Contributions

1. **CAFE**, a complete and reproducible pipeline for first-trimester loss prediction: 43
   engineered clinical features in five families, two-to-three view-specific experts, and a
   per-patient softmax gate, reaching AUROC 0.852 at a training cost of seconds rather than
   minutes.
2. **A clinically-grounded feature set with its contribution quantified per learner.**
   Rather than reporting that engineered features "help", we measure how much they help each
   of four learners and show the benefit is inversely proportional to the learner's native
   capacity to represent interactions — evidence that the features re-express existing signal
   rather than adding new signal.
3. **An evaluation that goes beyond discrimination**: Brier score, Cox calibration slope and
   intercept, expected calibration error, decision-curve analysis against treat-all and
   treat-none, and a screening operating point whose threshold is selected on one
   cross-validation repeat and scored on the other.
4. **A full interpretability stack**: exact out-of-fold TreeSHAP attributions, so that every
   patient is explained by a model that never trained on them; the gate's own per-patient
   trust weights; what those weights respond to; and worked high- and low-risk cases.
5. **A component-level ablation** decomposing the final AUROC into the contributions of the
   features, the boosting component, the gating architecture, and the sequence encoder. This
   is the central result of the paper.
6. **A cost analysis** measured on the same machine and the same folds as the accuracy
   numbers, so the accuracy-per-second trade-off is stated rather than assumed.

### 1.5 On novelty claims — what we do and do not assert

This work sits downstream of a proposal that framed its contributions as several
"first-ever" results for early pregnancy loss. We deliberately make no priority claims, for
two reasons, and we state them here rather than burying them in the limitations.

**First, we cannot verify priority.** Establishing that no prior work has engineered
discordance features or applied a gate to this outcome would require an exhaustive search we
have not performed, and a reviewer who knows of a single counterexample is entitled to
discard the claim — and to distrust the rest of the paper along with it. The contributions
above are stated as things we did and measured, not as things nobody has done before.

**Second, and more fundamentally, the data are synthetic.** No claim of the form "the first
clinical framework for early pregnancy loss" is available to a study with no patients in it.
What we contribute is methodological: a pipeline, an evaluation protocol, and an ablation
that identifies which of its parts carry the performance. Whether the engineered features
have clinical value is a question this cohort is structurally incapable of answering, and we
do not answer it.

---

## 2. Related Work

### 2.1 Engineered features against learned representations on tabular clinical data

That gradient-boosted trees remain competitive with, and frequently superior to, deep
architectures on tabular data is now well established [1, 2]. The mechanisms proposed —
rotation invariance, robustness to uninformative features, and an inductive bias toward
axis-aligned splits — all apply directly to the setting here, where the patient
representation is a few dozen derived scalars.

The implication for feature engineering is less often stated. A learner that already
discovers multiplicative and threshold interactions natively should benefit *less* from
having those interactions supplied explicitly than a learner that cannot. We test that
implication directly in §4.2 and use it as the interpretive key to the entire
feature-engineering result.

### 2.2 Gated fusion and mixtures of experts

Per-patient gating descends from the mixture-of-experts formulation [3], in which a gating
network learns an input-dependent distribution over specialist predictors, and from its
modern sparse variants [4]. In clinical multimodal work the gate is usually motivated by
*heterogeneity*: different patients have different data quality, or their risk is driven by
different mechanisms, and a fixed combination rule cannot adapt to that.

The motivation is sound and the mechanism is real. What is rarely reported is the
counterfactual — the same experts combined by a fixed rule, or dispensed with entirely in
favour of a single joint model — which is precisely what determines whether the adaptivity
was worth its cost. We therefore report the gate's *behaviour* and its *benefit* as two
separate measurements (§4.8), because they turn out to have different answers.

### 2.3 Evaluation beyond discrimination

A model's AUROC describes only its ranking of patients. Two models with identical AUROC can
differ substantially in whether their stated probabilities are trustworthy, and a
well-ranking, badly-calibrated model is actively unsafe when its output is used as a risk
figure in a consultation [5]. Decision-curve analysis [6] adds the further question of
whether acting on the model beats the default strategies of treating everyone or no one,
across the range of risk thresholds a clinician might plausibly adopt. Frameworks for
assessing prediction models have long recommended reporting discrimination, calibration and
clinical usefulness together rather than singly [12].

Because these are distinct properties, a tie in discrimination does not imply a tie in
clinical utility. That is why we measured them here rather than assuming the AUROC result
had settled the matter — and, as §4.4 shows, it is the reason this study reached a
conclusion its discrimination results alone could not have supported.

### 2.4 Early pregnancy-loss prediction

Prediction of first-trimester loss from early sonographic biometry and maternal history has
been approached both with classical risk scoring and, more recently, with deep models
operating on ultrasound recordings [7].

We do not compare our AUROC against figures reported on other cohorts. Prevalence, case mix
and measurement protocol differ, and a numerical comparison across cohorts would be
uninterpretable regardless of which direction it happened to favour. Published work enters
this paper as a source of *designs*, not of *results*.

---

## 3. Methods

### 3.1 Study design

The study proceeded in four pre-planned phases, each answering one question and each
reported separately below:

1. **Phase 1 — Features.** Do clinically-aligned engineered features improve prediction, and
   for which learners? (§3.3, §4.2)
2. **Phase 2 — Architecture.** Do view-specific experts combined by a per-patient gate
   improve on a single joint model given the same features? (§3.4, §4.1, §4.3)
3. **Phase 3 — Clinical evaluation.** Beyond discrimination, do the arms differ in
   calibration, net benefit, or behaviour at a screening operating point? (§3.6, §4.4–§4.6)
4. **Phase 4 — Interpretability.** What does the model rely on, what does the gate do, and
   does either correspond to what the components contributed? (§3.8, §4.7–§4.8)

Phases 1 and 2 are ablations under a fixed evaluation protocol; Phases 3 and 4 characterise
the resulting system. The decomposition in §4.10 assembles Phases 1 and 2 into a single
attribution of the final score.

### 3.2 Cohort

#### 3.2.1 Provenance and difficulty calibration

All data are synthetic, generated on the authors' machine. **No real patient data were used
at any stage.** The evaluation cohort (n = 3000, prevalence 20.3 %) is the same cohort used
throughout our fusion-architecture benchmark, generated from a single seeded command with
frozen parameters and reproducible bit-for-bit.

Its difficulty was calibrated *before* any of the models in this paper existed, and
calibrated against a baseline that is **not** one of the models under test
(HistGradientBoosting), to a band fixed in advance ([0.83, 0.87]; achieved 0.836 ± 0.013).
This ordering matters: calibrating difficulty against a proposed model allows the task to be
steered toward it, whereas using an external baseline prevents that.

#### 3.2.2 Modalities and measurement model

The scan modality comprises two to five visits per patient (mean 3.05), irregularly spaced
at 1–4 gestational-week intervals, with four measurements per visit: crown–rump length
(CRL), mean gestational-sac diameter (GS), yolk-sac diameter (YSD), and fetal heart rate
(FHR). The maternal modality comprises static clinical covariates: age, BMI, gravidity,
parity, previous loss, conception route, and plurality.

Marker trajectories are anchored to published first-trimester reference values across
gestational weeks 5–10, with marker-specific measurement error mimicking real ultrasound
precision (CRL ± 1.6 mm, GS ± 3.2 mm, FHR ± 6.5 bpm, YSD ± 0.55 mm) and a label-independent
constitutional-size factor correlating the size markers within a patient. Losses are
expressed as clinically recognised abnormality patterns — small CRL and sac, bradycardia,
and an enlarged yolk sac — each at 0.90–0.94 probability, so that no single marker is
deterministic.

#### 3.2.3 Outcome structure

The outcome has explicit causal structure: `label = label_sono OR label_mat`, verified at
1.0000 across all 3000 patients, with two independent channels — a sonographic channel
expressed in the scan trajectory and a maternal channel expressed in the static covariates.
Measured oracle ceilings are 0.820 for the sonographic channel alone, 0.756 for the maternal
channel alone, and 1.000 for both.

The channel labels are retained for diagnostics only. They are **never model inputs and
never evaluation targets**: because the observed label is their exact OR, supplying them as
features would leak the outcome outright.

#### 3.2.4 What is specified rather than estimated

The *strengths* of the two causal channels are specified parameters, not measured effect
sizes, and the maternal channel is near-deterministic by construction. The reference curves
make the *measurements* realistic; they do not make the *etiologic structure* an empirical
finding. No parameter of this cohort should be cited as a clinical quantity.

The cohort is nevertheless dual-etiology **by design**, built so that complementary signal
genuinely exists for fusion to exploit. A null result here is informative precisely because
the data were constructed to favour fusion.

### 3.3 Phase 1 — five families of clinically-aligned features

The engineered block adds 43 features to the 23-feature base table, giving a union view of
66 features. It is **label-free by construction**: the outcome column is never read during
feature construction, so no leakage pathway exists.

The organising idea is gestational normalisation. A raw crown–rump length of 20 mm is
meaningless without knowing the gestational age at which it was measured; the clinically
meaningful quantity is whether the measurement is normal *for that week*. Every measurement
is therefore converted to a z-score within its rounded gestational-age bin before any
further feature is derived from it, and all five families are built on those z-scores.

> **[TABLE 1 HERE] — Table 1.** The five feature families: the clinical question each
> encodes, its construction, and the number of features contributed.

Discordance is the family with the clearest clinical motivation and the least precedent. A
fetus whose heart rate is falling while its crown–rump length is still tracking normally is
in a different state from one where both are falling together, and neither raw marker
captures that. The divergence between them does.

### 3.4 Phase 2 — three encoders over three views

#### 3.4.1 Views and experts

CAFE partitions the available information into views and assigns an expert to each. All
experts see data derived from **both** modalities; the partition is by *representation*, not
by modality, so no expert is starved of a data source.

> **[TABLE 2 HERE] — Table 2.** Encoder specifications: the view each encoder consumes, its
> model class, and its parameter count.

Encoder B is not a new architecture. It is the same temporal-attention model used as a
full-size specialist elsewhere in this programme, scaled down from 88,193 parameters to
12,585 — matching the ~10,000-parameter budget proposed for a "light" sequence encoder.
Reporting it as a distinct design would overstate the novelty; it is a width-and-depth
ablation of a model we had already built.

#### 3.4.2 The gate

The gate is a two-layer MLP over the patient's own base features, emitting a softmax
distribution over the experts. The fused prediction is the gate-weighted sum of the experts'
**logits**, so the output remains on the log-odds scale and its sigmoid is a genuine pooled
probability rather than an arbitrary score. That property is what makes the calibration
analysis of §4.4 meaningful.

#### 3.4.3 Two implementation safeguards

Two details materially affect the result. Both are stated because both are easy to get
wrong, and getting either wrong inflates the architecture's apparent value.

**The gate trains on inner out-of-fold expert logits.** Within each training fold, expert
logits are produced by an inner 3-fold cross-validation; the experts are then refitted on
the full training fold for test-time prediction. Training the gate on in-fold logits would
teach it to trust whichever expert overfits hardest.

**Expert logits are placed on a common scale before gating.** In the three-expert
configuration we found that an unstandardised gate placed 0.697 of its mass on the *weakest*
expert — not because that expert was informative, but because its logits had the widest
spread, so it dominated the weighted sum at any weight. We therefore standardise each
expert's logits on training-fold statistics before gating. This is a genuine confound rather
than a tuning detail: on the diagnostic fold where we found it, correcting it moved the
sequence encoder's gate weight from 0.697 to 0.257 and the three-expert arm from 0.810 to
0.834. Left uncorrected, it would have been reported as an architectural finding about
sequence encoders when it was an artefact of unnormalised logit scale.

### 3.5 Baselines and input parity

Every baseline receives the **union view** — the base table plus all 43 engineered features
— so the hybrid's experts and the single-model baselines see exactly the same information.
The hybrid splits that union and gates it; the baselines get it whole. Nothing is withheld
from a baseline to flatter the hybrid, and no hybrid result can be attributed to an
information advantage.

Baselines are logistic regression (standardised), XGBoost [10], and CatBoost [11], all at
library defaults with a fixed seed. The base feature table is constructed through the
project's sanctioned feature-table function rather than rebuilt ad hoc; ad-hoc
reconstruction understates the tree baseline by approximately 0.019 AUROC and has twice
produced a spurious apparent win in this research programme.

Input parity is additionally verified by gradient and perturbation checks asserting that
each model's output actually depends on every modality it is credited with. That check was
introduced after a cross-attention wiring fault was found in which a softmax over a single
key produced identically zero gradient to the maternal pathway, silently making one tower
single-modality while results tables described it otherwise. **Parity is a property of the
wiring, not of the command-line flags.**

### 3.6 Evaluation protocol

All arms are evaluated with **5×2 repeated stratified cross-validation**
(`RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)`), on folds identical to
those used by every other study in this programme, so numbers here are directly comparable
to numbers published elsewhere from this cohort. Plain 5-fold cross-validation is unstable
at these effect sizes — one effect in this project halved from +0.012 to +0.003 when moved
from 5-fold to 5×2 — and single-split estimates are used nowhere in this paper.

Reported metrics are AUROC, AUPRC, Brier score, Cox calibration slope and intercept,
expected calibration error over ten equal-width bins, net benefit across risk thresholds
0.05–0.50, and sensitivity, specificity, PPV and NPV at a screening operating point [12].
**Bare accuracy is never reported**: at 20.3 % prevalence, predicting "no loss" for every
patient scores 79.7 %.

Repeated cross-validation yields two complete out-of-fold prediction vectors, one per
repeat. The screening threshold is selected on repeat 1, at the lowest value achieving 90 %
sensitivity, and then **scored on repeat 2**, so the operating point is never chosen and
reported on the same predictions.

### 3.7 The decision rule

Fixed in advance and applied verbatim to every comparison in this paper, including those
where its verdict is inconvenient. A difference between two arms is **REAL** only if:

> |mean fold-wise difference| **> 2 × pooled fold standard deviation**, **and** the sign of
> the difference agrees on **≥ 8 of 10 folds**.

Otherwise the result is reported as a **TIE**, regardless of the sign or size of the point
estimate. Reporting a favourable point estimate as a win when it fails this rule is the
specific failure mode the rule exists to prevent. The rule is applied symmetrically: several
of the ties below have point estimates that would be attractive to report as wins.

### 3.8 Phase 4 — interpretability protocol

Attributions are **exact TreeSHAP** [8, 9], computed by the boosting libraries themselves
rather than by approximation, and computed **out of fold**: the five folds of repeat 1 form
one complete partition, so every patient's attribution comes from a model that never saw
that patient in training. In-fold attributions describe a model's memorisation as readily as
its reasoning.

Gate behaviour is characterised by the distribution of its per-patient weights across the
out-of-fold cohort, by the correlation of those weights with patient features, and by worked
examples at the extremes of predicted risk.

### 3.9 Reproducibility

The cohort is regenerable bit-for-bit from one seeded command with frozen generator
parameters. Every table in this paper is produced by a named script from per-fold prediction
files that are retained, and every number in the manuscript is checked against those files
by an automated verifier that fails on any drift. Paths are listed under *Data and Code
Availability*.

---

## 4. Results

### 4.1 Discrimination and overall error

> **[TABLE 3 HERE] — Table 3.** Discrimination, overall error and calibration, 5×2 repeated
> CV, mean over 10 folds. All arms receive the identical 66-feature union view. CAFE denotes
> the gated two-expert configuration (Encoders A and C); the suffix indicates the boosting
> component from which both experts are built.

CAFE records the highest AUROC point estimate in the study. Against its matched single-model
comparator — CatBoost on the union of the same features — the difference is **+0.0008
AUROC, with sign agreement 6/10**, which the decision rule returns as a **TIE**. On AUPRC
the ordering reverses (0.686 against 0.692) and on Brier score it reverses again (0.1048
against 0.1042), both also ties. On every discrimination and overall-error measure we
computed, the two models are the same model.

That statement should be read precisely. It does not say CAFE is worse. It says the
architecture that splits the feature space into views and gates them recovers exactly what a
single tree recovers from the same features, and that we have no evidence, at this sample
size, of a difference in either direction.

### 4.2 Phase 1 — who actually benefits from engineered clinical features?

The 43 engineered features were evaluated by adding them to the base table for four learners
independently, holding folds, seeds and everything else fixed.

> **[TABLE 4 HERE] — Table 4.** Contribution of the engineered feature block, by learner:
> base table versus union view, with the fold-wise sign agreement and decision-rule verdict.

Every row is a tie under the decision rule. But the *ordering* is the finding, and it is
perfectly monotone: **the benefit of the engineered features is inversely proportional to
the learner's native capacity to represent interactions.** Logistic regression, which is
additive in its inputs and can represent no interaction unless one is handed to it, gains
the most (+0.015) and gains it with 10/10 sign consistency. The gradient-boosted trees,
which discover threshold interactions natively, gain from little (+0.005 for
HistGradientBoosting, +0.002 for CatBoost) to nothing at all (+0.000 for XGBoost, sign 4/10).

This is the interpretive key to the whole feature-engineering result. Features that supplied
genuinely *new* information — a measurement not previously available, a channel not
previously encoded — would help every learner. Features that *re-express* information
already present help only those learners that could not extract it themselves. The
engineered block behaves unambiguously like the second kind. It is an excellent
representation; it is not additional evidence.

Section 4.7 supplies the complementary observation from the opposite direction: the model
*uses* these features heavily. The two facts together are consistent only with re-expression.

### 4.3 Phase 2 — does the sequence encoder earn its place?

Encoder B was added as a third expert with the tree component held fixed, so the difference
isolates the sequence encoder.

> **[TABLE 5 HERE] — Table 5.** Encoder B ablation. Encoder B is the temporal-attention
> model at 12,585 parameters; all arms use the standardised-logit gate of §3.4.3.

Adding the sequence encoder does not help under either component, and both point estimates
are negative (−0.005 with CatBoost experts, −0.003 with XGBoost experts). Encoder B is not
useless in isolation — at 0.813 it is a competent standalone model, within noise of XGBoost
on the full union view — but it contributes nothing that the tree experts have not already
extracted from the same visits.

The gate reaches the same conclusion independently. Given three experts and no instruction
about which to prefer, it assigns Encoder B a mean weight of **0.195** against the 0.333 it
would receive under an equal split, in every fold. The architecture, left to its own
devices, learns to discount the component we added.

### 4.4 Calibration — the one place the arms genuinely separate

Discrimination ties everywhere. Calibration does not.

The Cox calibration slope measures whether predicted log-odds are correctly scaled: 1.0 is
correct, and below 1.0 means the model is over-confident, pushing risks too far toward 0 and
1. The CatBoost-derived arms sit at 0.923 and 0.946. The XGBoost-derived arms sit at 0.430
and 0.480 — their predicted log-odds are roughly twice as extreme as the data support.
Expected calibration error tracks it: 0.030–0.037 for the CatBoost family against
0.082–0.093 for the XGBoost family.

The per-fold distributions **do not overlap at all**. XGBoost's worst-to-best calibration
slope across ten folds spans 0.310–0.495; CatBoost's spans 0.684–1.061.

> **[FIGURE 1 HERE] — Figure 1.** Reliability curves, pooled out-of-fold. The two
> XGBoost-derived arms bow away from the diagonal in the same direction; the
> CatBoost-derived arms track it. *(Source figure: figures/cafe/fig_cafe_calibration.png)*

The practical consequence runs opposite to the paper's other findings. Two arms that are
indistinguishable on AUROC — XGBoost at 0.830 and CatBoost at 0.851 are only 0.021 apart —
differ by a factor of two in whether their stated probabilities can be believed. **A model
selected on AUROC alone would not have detected this**, and a 12 % predicted risk from the
XGBoost arm does not mean what a 12 % predicted risk from the CatBoost arm means.

Between CAFE and its matched CatBoost comparator, calibration is a tie under both
formulations of the rule discussed next (slope Δ = +0.023, 8/10; ECE Δ = −0.007, 8/10).

#### 4.4.1 A defect in our own decision rule, reported rather than resolved silently

Applied literally, our rule returns TIE for the XGBoost-versus-CatBoost calibration
comparison (Δ = −0.493, 2 × pooled SD = 0.520, sign 10/10). The reason is a defect in the
rule's *implementation* that this is the first comparison in the project to expose: the
pooled standard deviation is computed by concatenating both arms' fold values, so when two
arms are far apart the "pooled SD" absorbs the very gap being tested and inflates the
threshold. Computing the pooled *within-arm* standard deviation instead — the quantity the
rule's wording actually describes — gives 2 × SD = 0.175 against Δ = −0.493 with 10/10 sign
agreement, a **REAL** difference.

We have verified that this distinction changes nothing previously reported. Across all 56
pairwise AUROC comparisons in this study for which per-fold values are retained, **the two
formulations return identical verdicts in every case**; the ratio of concatenated to
within-arm SD is 0.95–1.04 for AUROC and 2.97 for the calibration slope. The rule is sound
in the regime it was written for, where arms differ by less than their fold-to-fold noise,
and misbehaves only where a difference is large — which had never previously occurred.

We note finally that the comparison that flips is between two *baselines*, and that it
reflects badly on the component originally proposed for this hybrid. No claim of ours is
advanced by reporting it.

### 4.5 Clinical utility — decision-curve analysis

> **[TABLE 6 HERE] — Table 6.** Net benefit at representative risk thresholds, pooled
> out-of-fold (repeat 2), against the treat-all and treat-none default strategies.

Mean net benefit above the better of the two default strategies, across thresholds
0.05–0.50, is CAFE **+0.0744**, CatBoost +0.0737, logistic regression +0.0648, CAFE-XGB
+0.0626, and XGBoost +0.0617.

Two things follow. First, **every model is substantially more useful than either default
strategy** across the entire clinically plausible threshold range. At a 20 % threshold,
treating everyone yields a net benefit of 0.0042 — essentially nothing, because at 20.3 %
prevalence treat-all sits at its break-even point — while every model yields 0.10 to 0.11.
This is the paper's clearest positive result, and it is a property of the *task and the
features* rather than of any architecture, since all five arms achieve it.

Second, the arms again do not separate from one another. The spread between the best and
worst arm's mean net benefit (0.0128) is smaller than the gap between the worst arm and the
default strategies (0.0617).

> **[FIGURE 2 HERE] — Figure 2.** Net benefit against risk threshold, with treat-all and
> treat-none references. The five model curves are visually superimposed; all sit far above
> both defaults beyond a 15 % threshold. *(Source figure:
> figures/cafe/fig_cafe_decision_curve.png)*

### 4.6 The screening operating point

> **[TABLE 7 HERE] — Table 7.** Performance at a 90 %-sensitivity screening threshold. The
> threshold is selected on repeat 1 and scored on repeat 2, so it is never chosen and
> reported on the same predictions.

At matched sensitivity near 90 %, CAFE flags 53.9 % of the cohort against XGBoost's 62.3 %
— 8.4 percentage points fewer patients referred for the same case detection. NPV is 0.955:
of the patients the model clears, roughly 1 in 22 nonetheless experiences loss, which sets a
firm ceiling on how such an output could be used in practice.

**This table carries a weaker warrant than the others in this paper**, and we flag that
explicitly: it is a single split-derived comparison rather than a ten-fold estimate, so the
decision rule cannot be applied to it and the ordering should not be treated as established.
Note also that the selected thresholds differ by an order of magnitude across arms, from
0.004 to 0.081 — a direct consequence of the calibration differences in §4.4, and a reminder
that a threshold tuned for one model is meaningless when transferred to another.

### 4.7 Attribution — the engineered features dominate the explanation

> **[TABLE 8 HERE] — Table 8.** Top 15 features by mean |SHAP|, out-of-fold exact TreeSHAP
> over all 3000 patients, with the view each feature belongs to.

> **[FIGURE 3 HERE] — Figure 3.** Top 20 features by mean |SHAP|, coloured by view.
> Engineered features occupy 13 of the top 20 places. *(Source figure:
> figures/cafe/fig_cafe_shap.png)*

The engineered block carries **66.8 %** of total attribution mass and supplies 13 of the top
20 features. The discordance feature `disc_last_CRL_YSD` — the gap between crown–rump length
and yolk-sac diameter, each normalised for gestational age — is the third most important
feature in the model, and the maternal composite is second.

**This is the sharpest illustration of the paper's central distinction.** By any
attribution-based reading, the engineered features are what the model is using. By the
controlled ablation of §4.2, removing them costs CatBoost 0.002 AUROC. Both are true, and
they are compatible only if the features are a *re-expression* of signal the model could
otherwise reach by a different route. **Feature importance measures what a model relies on
given the features it has; it does not measure what those features contributed over not
having them.** Reporting the SHAP table without the ablation would have made a compelling
and entirely unfounded case for the feature engineering.

Three high-risk and three low-risk out-of-fold cases are reported in the released case
file. The highest-risk patient (predicted 0.998, true label loss) is driven by
`mat_composite` = 4.02 (SHAP +1.23), age 40.7 (+0.91) and `disc_last_CRL_YSD` = −1.73
(+0.73) — a legible chain of reasoning combining maternal risk with sonographic discordance.
We also report a failure: patient `P00310` receives a predicted risk of 0.003 and
nonetheless experienced loss, with the model's confidence driven by a low BMI of 16.0 and
favourable maternal terms. Individual attributions explain what the model did; they do not
certify that it was right.

### 4.8 The gate is confidently per-patient, and it does not matter

A gate that collapses to a near-constant weight is a fixed blend wearing a gate's clothing.
This one does not collapse.

Across out-of-fold patients the weight on Encoder A has **mean 0.557 and standard deviation
0.402, spanning the full range from 0.000 to 1.000**. The distribution is close to bimodal:
55.9 % of patients are routed predominantly to the flat expert and 44.1 % predominantly to
the cross-modal expert. The gate is making confident, patient-specific routing decisions.

Its policy is also clinically legible. The weight correlates most strongly and negatively
with the yolk-sac measurements — `zYSD_last` (−0.309), `zYSD_mean` (−0.285), `YSD_latest`
(−0.285) — and positively with maternal gravidity (+0.220) and age (+0.217). In plain terms:
**when the yolk sac looks abnormal, the gate routes the patient to the cross-modal expert;
when the patient's maternal risk profile is prominent, it routes to the flat expert.** That
is a sensible policy, and nobody specified it.

It is worth **+0.001 AUROC** (§4.1).

We consider this the most interesting single result in the paper. The usual explanation for
a gate failing to help is that it failed to learn anything — that it degenerated to an
average. That explanation is unavailable here. This gate learned a confident, coherent,
interpretable routing policy, and the policy made no difference to any outcome measure,
because both experts were already extracting substantially the same signal from the same
patients. **Gate behaviour and gate benefit are separate measurements, and reporting only
the first would have been evidence for a conclusion the second contradicts.**

### 4.9 Cost

> **[TABLE 9 HERE] — Table 9.** Training and inference cost, measured on one fold on the
> same machine as all accuracy results. Deep-model figures are from the same programme's
> runtime benchmark on the same cohort. Inference for the CAFE arms is a component sum — two
> tree evaluations plus a 418-parameter MLP forward pass — rather than a direct end-to-end
> measurement, and is marked as an estimate.

Against the deep architectures, CAFE-XGB trains **75×** faster and CAFE-CatBoost **6.2×**
faster than the FETA-Transformer, at roughly 100× lower inference cost. Against its own
matched baseline the direction reverses: CAFE-CatBoost costs **5.3×** a single CatBoost fit,
and CAFE-XGB **10.6×** a single XGBoost fit, for +0.001 and +0.003 AUROC respectively.

The gate itself is 418 parameters. Whatever CAFE's merits, parameter economy relative to the
deep models is not principally an achievement of the fusion design — it is an achievement of
using trees.

> **[FIGURE 4 HERE] — Figure 4.** AUROC against training time on a logarithmic axis. The
> frontier is flat: two orders of magnitude of additional compute buy approximately 0.02
> AUROC, and the last 5× buys 0.001. *(Source figure:
> figures/cafe/fig_cafe_accuracy_vs_cost.png)*

### 4.10 Where the accuracy actually comes from

Assembling the controlled ablations into a single path from the simplest model to the
proposed one, each step changing exactly one thing:

> **[TABLE 10 HERE] — Table 10.** Decomposition of the final AUROC: a five-step path from
> logistic regression on the base table to the full CAFE system, each step altering exactly
> one component.

> **[FIGURE 5 HERE] — Figure 5.** The same decomposition rendered as a waterfall. *(Source
> figure: figures/cafe/fig_cafe_ablation.png)*

The total journey from step 0 to CAFE is **+0.030** AUROC. Of that, **+0.021 is the choice
of boosting library** and **+0.015 is the engineered features** — but the latter only for a
learner weak enough to need them, since by step 3 the same features are worth +0.002. **The
fusion architecture that the system is named for contributes +0.001**, and the sequence
encoder contributes −0.005.

Every individual step in this table is a tie under the decision rule. The decomposition is
nonetheless informative, because the *relative* magnitudes are stable and consistently
signed, and because they indicate where effort should go: on this problem, choosing the
component and representing the features well dominates designing the fusion mechanism by
more than an order of magnitude.

---

## 5. Discussion

### 5.1 What CAFE delivers

CAFE is a good system and we would deploy it. It attains the highest AUROC point estimate
recorded in this research programme (0.852), is the best-calibrated arm we measured (slope
0.946, ECE 0.030) with no post-hoc correction applied, produces the highest mean net benefit
across clinically plausible thresholds, flags 8.4 percentage points fewer patients than the
XGBoost arm at matched 90 % sensitivity, trains in 43 seconds, predicts in microseconds, and
emits both a per-feature attribution and a per-encoder trust weight for every individual
patient. Against the deep fusion architectures this line of work began with, it is 6–75×
cheaper to train and roughly 100× cheaper at inference, and it is not less accurate than any
of them.

Each of those claims is measured on identical folds against baselines given the same
features and the same budget. None of them requires the reader to accept that the fusion
architecture is what produced the result — which is fortunate, because it is not.

### 5.2 The architecture is not the active ingredient

The central finding is the decomposition of §4.10. Removing CAFE's defining feature — the
partition into views and the learned per-patient gate — and replacing it with a single tree
on the union of the same features costs +0.001 AUROC, against a fold-to-fold standard
deviation of 0.022. The engineered features contribute meaningfully only to learners that
cannot represent interactions natively. The sequence encoder contributes negatively. The
largest single contributor is the boosting library.

We want to be precise about what this does and does not license. It does **not** show that
gated multi-view fusion cannot work, in general or even on this problem with more data. It
shows that on this cohort, at this sample size, with these features, the fusion mechanism
recovers what a single joint model already recovers — and that a study reporting CAFE's
0.852 without the ablation would have attributed the number to the wrong component.

The distinction matters practically rather than philosophically. A team that believes the
gate is the active ingredient will invest in better gates. On this evidence they should
invest in better features for weak learners, in better component selection, and — per §4.4 —
in calibration.

### 5.3 Why measuring beyond AUROC changed the conclusion

Had this study stopped at discrimination, its finding would have been "all five arms tie",
and the choice between them would have looked arbitrary. Calibration produced the one
genuine, consistently-signed, non-overlapping separation in the entire study: the
XGBoost-derived arms have calibration slopes near 0.45 against the CatBoost-derived arms'
0.93 — a factor of two in whether their probabilities can be believed, invisible to AUROC
and nearly invisible to Brier score, which conflates calibration with discrimination and
returned a tie.

This has a direct consequence for the operating-point table of §4.6. The thresholds
achieving 90 % sensitivity range from 0.004 for XGBoost to 0.081 for logistic regression, a
twenty-fold spread driven almost entirely by calibration rather than by any difference in
ranking. Any workflow that hard-codes a risk threshold — "refer if predicted risk exceeds
15 %" — is silently coupled to the calibration of the specific model that produced it.

The general lesson is that model-selection criteria determine which differences a study is
capable of detecting. This study was designed to detect architectural differences and found
none; the difference it did find lay in a dimension nobody had proposed to look at.

### 5.4 Relation to our fusion-architecture benchmark

This paper is the constructive complement to a pre-registered benchmark from the same
programme, in which eight architecturally distinct fusion families proved statistically
indistinguishable on this cohort (AUROC 0.822–0.840, a total spread smaller than any single
arm's fold standard deviation). That study asked whether *any* fusion architecture separates
from the others and concluded no. This one asks whether a carefully engineered,
clinically-grounded, compute-efficient hybrid can be built that performs well and explains
itself, and concludes yes — while locating its performance in its components rather than in
its fusion.

The two results are consistent and mutually reinforcing. Nothing here overturns the earlier
null; CAFE's 0.852 against CatBoost's 0.851 is another instance of it, obtained by a
different route with a different architecture and a different feature set. We note in
particular that the earlier study's verdicts are **unaffected** by the decision-rule defect
identified in §4.4.1: we re-checked every pairwise AUROC comparison for which per-fold
values are retained and found identical verdicts under both formulations.

### 5.5 Limitations

**The data are synthetic.** This is the binding limitation and it constrains every clinical
reading of this work. No patients were involved. The causal channel strengths are specified
parameters rather than measured effect sizes, and the maternal channel is near-deterministic
by construction — which is very likely why `age` and `mat_composite` dominate the
attribution table, and means that table should be read as a description of our generator,
not of miscarriage. Whether the discordance features have clinical value is a question this
cohort is structurally incapable of answering.

**The engineered features were designed by us against a generator whose structure we know.**
We did not use the label during feature construction and we verified this, but we cannot
claim to have been blind to the data-generating process. On real data the feature block
might do better, or considerably worse.

**A single cohort at n = 3000.** All confidence intervals here are wide relative to the
effects discussed. Several ties would remain ties at any sample size we could plausibly
reach, but the +0.021 attributed to the boosting component is a point estimate from ten
folds of one cohort and should not be read as a general ranking of boosting libraries.

**Encoder B was tested at one size.** A 12,585-parameter temporal attention model does not
help. We did not sweep its width, depth, or training budget beyond matching the neural
budget used elsewhere in the programme, so "sequence encoders do not help here" is supported
only at the scale tested — though we note that the full-size 88,193-parameter version,
tested separately, also did not help.

**The operating-point table rests on a single split**, and no decision-rule verdict is
available for it, as flagged in §4.6.

**We report per-patient attributions, not causal explanations.** SHAP values describe the
model's use of its inputs. They are not statements about mechanism, and the false-negative
case in §4.7 is a confident, well-attributed, wrong prediction.

### 5.6 Recommendations for practice

For practitioners building multimodal clinical predictors in this regime — short sequences,
small covariate panels, a few thousand patients:

1. **Establish the single-joint-model baseline on the union of your features before building
   any fusion architecture**, and give it the same feature engineering and the same tuning
   budget. On this problem it was never beaten.
2. **Report calibration alongside discrimination.** It was the only dimension in this study
   that separated the arms, and it determines whether a fixed decision threshold means
   anything at all.
3. **Do not infer feature value from feature importance.** Our engineered block carries 67 %
   of attribution mass and contributes 0.002 AUROC to the deployed learner. Only an ablation
   answers the question that attribution appears to answer.
4. **Measure gate behaviour and gate benefit separately.** A gate can be confident,
   patient-specific, and clinically legible while contributing nothing.
5. **Report compute.** It was the only axis on which the deep architectures in this
   programme differed from the simple ones, and it differed by two orders of magnitude.

### 5.7 Future work

Three directions follow from these results, and one does not.

**Change the shape of the data, not its difficulty.** The tree ties every temporal
architecture here because a two-to-five-visit trajectory flattens without loss into a
fixed-width table. The informative experiment is therefore many more visits per patient, not
irregular timing — the present data are already irregularly spaced — and not an easier
cohort, on which every architecture saturates and nothing can be measured.

**Treat calibration as a first-class selection criterion.** This study found its only real
separation there, by accident rather than design. A study powered for calibration rather
than for discrimination would be a different and probably more useful study.

**Test the feature block on real data.** The features were designed against a generator whose
structure we knew. Their value on data we did not construct is genuinely open.

What does *not* follow is further investment in the gate. Its behaviour is already confident
and legible; the constraint is not the gate's quality but the absence of complementary signal
for it to route between.

---

## 6. Conclusion

We built the clinically-grounded hybrid that a review of our earlier negative result
recommended: five families of engineered features, expert encoders over separate views of
the patient, and a per-patient learned gate. It reaches AUROC 0.852 — the best point estimate
in this research programme and within the 0.840–0.855 band predicted for it — with good
calibration, favourable decision-curve behaviour, full per-patient interpretability, and
training times measured in seconds rather than minutes.

It also ties the single gradient-boosted tree it is built from, on every measure we
computed. A controlled decomposition attributes +0.021 AUROC to the choice of boosting
component, +0.015 to the engineered features *for a learner weak enough to need them* and
+0.002 for the one actually deployed, **+0.001 to the gated fusion architecture the system is
named for**, and −0.005 to the sequence encoder. The gate learns a confident, bimodal,
clinically legible routing policy — and that policy changes no outcome measure.

The system is worth having. The explanation for why it works is not the one its architecture
suggests, and we would rather publish the correct explanation of a good result than an
attractive explanation of the same result. The one dimension along which the arms genuinely
separated was calibration, which no part of the original proposal — including ours — had
identified as the place to look.

All data are synthetic; these findings concern modelling methodology and carry no clinical
implication for real pregnancies.

---

## Declarations

### Ethics approval and consent to participate

Not applicable. This study involved no human participants, no animal subjects, and no
identifiable data of any kind. All data analysed are synthetic and were generated
computationally by the authors. Institutional review board approval and informed consent
were therefore not required.

### Consent for publication

Not applicable. No individual person's data are reported. All patient identifiers appearing
in this manuscript denote synthetic records.

### Data and code availability

All code, generator settings, per-fold predictions and result tables are held in the project
repository. The evaluation cohort is reproducible bit-for-bit from a single seeded command.

| Artifact | Path |
|---|---|
| Feature construction (5 families, 43 features) | `results/xmodal_study_2026-07-22/xmodal_features.py` |
| Gated hybrid on stronger experts | `results/xmodal_study_2026-07-22/hybrid_better_experts.py` |
| Encoder B ablation | `results/xmodal_study_2026-07-22/encoder_b_lightattention.py` |
| Phase 3 + 4 clinical evaluation | `results/xmodal_study_2026-07-22/clinical_evaluation.py` |
| Figure generation | `results/xmodal_study_2026-07-22/make_cafe_figures.py` |
| Per-fold values for every table | `results/xmodal_study_2026-07-22/*_per_fold.csv` |
| Decision-rule verdicts | `results/xmodal_study_2026-07-22/*_verdicts.csv` |
| Worked case examples | `results/xmodal_study_2026-07-22/case_examples.csv` |
| Input-parity check | `scripts/check_input_parity.py` |
| Numerical verification of this manuscript | `results/xmodal_study_2026-07-22/verify_cafe_manuscript.py` |

### Reporting statement

This manuscript reports every configuration evaluated, including those that favoured the
proposed system before correction and those that did not. No configuration was excluded
after the fact.

Two implementation faults that would have produced misleading results are reported in the
text rather than silently corrected: the unnormalised expert-logit scale that caused the
gate to place 0.697 of its mass on the weakest expert (§3.4.3), and the
concatenated-standard-deviation term in the decision rule that returns TIE for a
non-overlapping difference (§4.4.1). In the latter case we report the rule's literal verdict
first, then the corrected one, and verify that no previously published verdict is affected.

The proposed system's headline number, AUROC 0.852, falls inside the 0.840–0.855 band that
was predicted for it before the experiments were run. We report that the prediction was
accurate and that the mechanism proposed to explain it was not.

The data are synthetic throughout. The channel strengths in the generator are specified
rather than estimated, and no parameter of this cohort should be cited as a clinical
quantity.

### Author contributions

**A. V. M. Kumaran:** conceptualisation, methodology, software, formal analysis,
investigation, data curation, visualisation, writing — original draft. **M. Sundara Rajan:**
conceptualisation, methodology, validation, supervision, writing — review and editing. Both
authors read and approved the final manuscript.

### Funding

This research received no specific grant from any funding agency in the public, commercial,
or not-for-profit sectors.

### Competing interests

The authors declare that they have no competing interests.

### Acknowledgements

We thank the reviewer of our earlier negative-result manuscript, whose specific
architectural proposal — engineered clinical features, view-specific experts, and a
per-patient gate — defined the system evaluated here and motivated the ablation that
followed.

---

## References

[1] Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need.
*Information Fusion*, 81, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011

[2] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still
outperform deep learning on typical tabular data? *Advances in Neural Information Processing
Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track.*

[3] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of
local experts. *Neural Computation*, 3(1), 79–87.

[4] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J.
(2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
*International Conference on Learning Representations (ICLR 2017).*

[5] Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W.
(2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230.

[6] Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for
evaluating prediction models. *Medical Decision Making*, 26(6), 565–574.

[7] Liu, L., Zang, Y., Zheng, H., Li, S., Song, Y., Feng, X., Zhang, X., Li, Y., Cao, L.,
Zhou, G., Dong, T., Huang, Q., Pan, T., Deng, J., & Cheng, D. (2025). An AI method to
predict pregnancy loss by extracting biological indicators from embryo ultrasound recordings
in early pregnancy. *Scientific Reports*, 15, 25946.
https://doi.org/10.1038/s41598-025-11518-5

[8] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).*

[9] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R.,
Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global
understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.

[10] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining*, 785–794.

[11] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).
CatBoost: unbiased boosting with categorical features. *Advances in Neural Information
Processing Systems 31 (NeurIPS 2018).*

[12] Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N.,
Pencina, M. J., & Kattan, M. W. (2010). Assessing the performance of prediction models: a
framework for traditional and novel measures. *Epidemiology*, 21(1), 128–138.
