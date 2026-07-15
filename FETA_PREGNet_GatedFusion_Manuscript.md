# Gated Multimodal Fusion of Temporal-Attention and Knowledge-Guided Graph Learning for First-Trimester Pregnancy-Loss Prediction: A Synthetic Dual-Etiology Cohort Study with an Input-Parity Evaluation Protocol

**Mr. A. V. M. Kumaran¹\*, Dr. M. Sundara Rajan²**

¹Research Scholar, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035. Email: avmprofessor@gmail.com
²Associate Professor, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035.

*\*Corresponding author:* Mr. A. V. M. Kumaran, Research Scholar, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035. Email: avmprofessor@gmail.com

---

## Abstract

First-trimester pregnancy loss affects roughly one in five clinically recognized pregnancies, and its prediction is complicated by the fact that losses arise from **heterogeneous etiologies** — some expressed primarily in the embryonic ultrasound trajectory, others driven primarily by maternal physiology. This heterogeneity motivates *multimodal fusion*: a model that reads the longitudinal scan sequence and the maternal profile jointly should recover losses that a single-modality view would miss. We investigate whether a purpose-built **gated fusion architecture** delivers a measurable advantage over simple, parameter-free alternatives when every model is held to strict input parity. We pair two complementary, interpretable towers — the **FETA-Transformer**, which applies gestational-age-aware temporal attention to the variable-length scan sequence, and **PREG-Net**, a hand-implemented knowledge-guided graph-attention network over clinical and ultrasound variables — and fuse them with a **HybridFusionPredictor** that exchanges information between the towers through masked cross-attention and then routes between their predictions with a *learned per-patient gate* plus a correction term. Because no consented patient data were available, all experiments use a fully synthetic cohort of **3,000 patients (9,098 scans; 18.8 % loss)** produced by a documented dual-etiology generator in which each loss is the logical OR of an independent sonographic channel and an independent maternal channel; we verify the OR structure holds for 100 % of rows and disclose that the channel strengths are **specified, not estimated**. Crucially, and unlike the common practice in fusion papers, we do **not** handicap the single-model baselines: under our **input-parity protocol** every model — including each standalone tower — sees both modalities, so any advantage of the hybrid must come from the *fusion mechanism itself* rather than from starving the baselines. On a single held-out split the gated hybrid ranks first (test AUROC 0.9975), ahead of a plain logit-average ensemble (0.9944) and a standalone PREG-Net given the full multimodal graph (0.9938). However, under five-fold cross-validation this ordering does not survive: the gated hybrid (0.988 ± 0.004) is **statistically indistinguishable from — and on average marginally behind — a parameter-free logit average of the same two towers (0.989 ± 0.006)**, winning only one of five folds, and a single PREG-Net that reads both modalities reaches 0.981 ± 0.006 on its own. A gradient-boosted tree over engineered features ties the hybrid (0.986 ± 0.004). We therefore report a deliberately candid result: on this cohort the multimodal *input* is highly informative and the two towers are genuinely complementary, but the **gated fusion architecture earns no measurable benefit over trivial averaging**. We also surface a concrete, unexploited architectural finding — the transformer tower systematically under-uses the maternal variables because of a late cross-attention bottleneck (AUROC 0.884 vs PREG-Net's 0.981 on identical inputs). This work is framed as a **methodological, reproducibility, and evaluation-protocol contribution**, not as clinical evidence.

**Keywords:** Early pregnancy loss, longitudinal ultrasound, transformer attention, graph attention networks, multimodal fusion, gated fusion, synthetic clinical data, honest evaluation, input parity

---

## 1. Introduction

### 1.1 Background

The first trimester is the interval in which the majority of clinically recognized pregnancy losses occur, affecting an estimated 10–20 % of recognized pregnancies. Transvaginal and transabdominal ultrasound provide the principal non-invasive window onto early embryonic development, and several quantitative markers acquired during routine first-trimester scanning carry prognostic value: fetal heart rate (FHR) indexes early cardiac function, crown–rump length (CRL) measures embryonic growth, and gestational-sac (GS) and yolk-sac (YSD) dimensions characterize the developing support structures. Maternal factors — age, body-mass index, parity, gravidity, previous loss, mode of conception, and plurality — modulate the baseline risk against which these ultrasound signals are read.

A biological fact that is central to this paper is that early loss is **etiologically heterogeneous**. Some losses are driven by an embryonic abnormality that is expressed in the scan trajectory (depressed or declining FHR, lagging growth); others are driven predominantly by maternal physiology and may show comparatively unremarkable early scans. A predictor that reads only one modality is therefore structurally blind to a fraction of losses, which is exactly the situation in which combining a temporal ultrasound view and a relational maternal view — *multimodal fusion* — should help.

### 1.2 Challenges and Motivation

Four recurring limitations motivate this work; the first three are shared with the prior literature and the fourth is specific to the fusion question we ask.

First, **static and threshold-based interpretation.** Much of clinical practice reduces markers such as FHR to fixed cut-offs, ignoring that embryonic cardiac activity rises steeply and non-linearly across the early weeks; a single value stripped of its gestational context and trajectory is an impoverished predictor.

Second, **loss of longitudinal structure.** Where multiple scans exist, they are frequently collapsed into scalar features (mean, slope, last value), discarding the ordering and spacing of observations — precisely the information needed to detect subtle week-to-week deviations.

Third, **implicit interaction modeling and opacity.** Feature interactions are typically captured implicitly (tree splits, black-box networks), exposing neither which variables drove a prediction nor which physiological relationships were implicated.

Fourth — and the specific target of this study — **fusion claims that are not honestly tested.** Papers that propose a multimodal fusion architecture routinely compare it only against the single-modality sub-models it fuses. When those sub-models are each restricted to one modality, the fusion model wins *by construction*, and the comparison says nothing about whether the fusion *mechanism* is worth its complexity. The honest tests — does the learned fusion beat a **parameter-free average** of the same towers, and does it beat a **single model given the same multimodal input?** — are frequently omitted.

### 1.3 Problem Statement

We ask a narrow, falsifiable question: **given identical multimodal inputs, does a gated fusion of a temporal-attention tower and a knowledge-graph tower measurably outperform (i) a parameter-free logit average of the same two towers, and (ii) a single strong model that reads the same inputs?** Answering this requires an evaluation protocol that forbids baseline handicapping and that reports every gap against its fold-to-fold uncertainty.

### 1.4 Research Gap

At the intersection of longitudinal temporal modeling, explicit relational modeling, and multimodal fusion for first-trimester loss prediction, no prior work has (a) evaluated a temporal-attention model and a knowledge-guided graph model head-to-head under a shared, parity-controlled protocol, (b) introduced a *gated per-patient* fusion of the two, and (c) subjected that fusion to an explicit "defensibility test" against trivial averaging and a same-input single model. This study does all three, on a controlled synthetic cohort in which the complementary structure that fusion is meant to exploit is present by construction and fully documented.

### 1.5 Objectives

The primary objective is to design, implement, and **honestly** evaluate a gated multimodal fusion architecture for first-trimester loss prediction under strict input parity. Secondary objectives are to: (a) model irregular longitudinal ultrasound sequences without hand-crafted temporal features; (b) encode physiological relationships explicitly through a knowledge-guided graph; (c) construct a documented dual-etiology synthetic cohort in which complementary sonographic and maternal signal exists by design; (d) test whether the gated fusion beats a parameter-free average and a same-input single model, reporting each gap with its cross-validated uncertainty; and (e) extract interpretable temporal- and graph-attention explanations. Because consented patient data were unavailable, a further explicit objective is to make the absence of real-world validation unambiguous.

### 1.6 Contributions

1. **A gated multimodal fusion architecture (HybridFusionPredictor).** The FETA-Transformer's per-scan tokens and PREG-Net's per-node embeddings exchange information through masked bidirectional cross-attention; a **learned per-patient gate** then routes between the two towers' predictions, with a small correction term and auxiliary per-branch losses that keep both towers individually predictive. This is an intermediate-fusion design, distinct from a late-fusion ensemble.

2. **A documented dual-etiology synthetic generator** in which each loss is the logical OR of an independent sonographic channel and an independent maternal channel, so that complementary modality signal — the very thing fusion is meant to exploit — is present *by construction* and its strength is *specified rather than estimated*.

3. **An input-parity evaluation protocol and an explicit "defensibility test."** Every model, including each standalone tower, receives both modalities; the fusion is judged not against handicapped sub-models but against a parameter-free logit average of the same towers and against a single same-input model, with every gap reported against its fold standard deviation.

4. **A candid result.** On this cohort the multimodal input is highly informative and the towers are complementary, yet the gated fusion architecture provides **no measurable advantage over trivial averaging** — a negative-leaning finding we report in full, together with the ceiling effects that explain it.

5. **A concrete architectural finding and interpretability analysis.** On identical inputs the transformer tower under-performs the graph tower by ~0.10 AUROC because its maternal pathway is a late cross-attention bottleneck; and both towers expose clinically legible attention-based explanations.

---

## 2. Related Work

Recent research on early-pregnancy outcome prediction increasingly integrates ultrasound imaging, structured clinical markers, and machine learning. We summarize representative studies and position the present work against them; Table 1 provides a structured comparison.

**AI-driven biometric extraction and ensemble prediction.** Liu et al. developed a two-stage framework in which a modified U-Net automatically measures gestational-sac area, YSD, CRL, and FHR from early-pregnancy ultrasound video, feeding these biometrics into an ensemble learner (CatBoost, AUC 0.969). This demonstrates that automated measurement can remove operator variability, but it treats the extracted biometrics as static tabular inputs and models neither their temporal trajectory nor their mutual dependence.

**Week-specific risk modeling.** In a recurrent-loss cohort, Liu et al. built week-wise models combining ultrasound indices, demographics, and serum markers, showing that informative predictors change with gestational week and that discrimination improves as pregnancy advances (AUC 0.671 → 0.872, weeks five to seven). This underscores the value of temporal, week-aware modeling but keeps the per-week models separate rather than integrating the trajectory in a single learner.

**Radiomics and multimodal fusion.** Yan et al. extracted radiomic features from multimodal transvaginal ultrasound for endometrial receptivity in unexplained recurrent loss (XGBoost combined model AUC 0.871/0.844; SHAP for explainability), and later fused a ResNet-50 image model with TabNet (fusion AUC 0.853). Zhang et al. used structured semantic features from two-dimensional ultrasound (test AUC up to 0.982). These confirm that multimodal fusion and explainability help, but they rely on image-derived features and do not model longitudinal biometrics as sequences or as an explicit relational graph — and, importantly for our question, they compare fusion primarily against their own sub-models rather than against parameter-free fusion.

**Clinical-radiomic fusion and preconception models.** Murugesu et al. combined radiomic and clinical features via elastic-net logistic regression across two hospitals (train AUC 0.91; external AUC 0.82), providing valuable external validation. Yang et al. built a preconception model for subsequent early loss (best AUC 0.805 train, 0.781 validation). These establish the clinical framing and the importance of calibration, external validation, and decision-curve analysis.

**Positioning of the present work.** The proposed framework differs in three respects. First, the FETA-Transformer *learns* temporal representations directly from irregularly spaced sequences instead of summarizing them into scalars. Second, PREG-Net encodes physiological relationships *explicitly* as a knowledge-guided graph and exposes node/edge reasoning. Third — and most distinctively — we ask whether the *fusion mechanism itself* is worth its complexity, and we answer that question under an input-parity protocol with an explicit defensibility test, on synthetic data whose complementary structure is documented. The study is framed as a reproducibility, methods, and evaluation contribution, with strong baselines reported honestly alongside the proposed models.

**Table 1. Comparative summary of related work and the present study.**

| Ref. | Data / setting | Method | Temporal modeling | Interaction modeling | Fusion tested honestly? | Reported performance |
|---|---|---|---|---|---|---|
| Liu et al. [15] | 630 ultrasound videos (6–10 wk) | U-Net + ensemble (CatBoost) | No (static biometrics) | Implicit (tree ensemble) | N/A | AUC 0.969 |
| Liu et al. [16] | RPL cohort; ultrasound + serum | Week-wise logistic regression | Per-week (separate models) | None | N/A | AUC 0.671–0.872 |
| Yan et al. [17] | 346 RPL vs 369 controls; multimodal TVUS | XGBoost + radiomics + SHAP | No | Implicit | vs sub-models only | AUC 0.871 / 0.844 |
| Yan et al. [18] | Ultrasound + clinical | ResNet-50 + TabNet fusion | No | Fusion | vs sub-models only | Fusion AUC 0.853 |
| Zhang et al. [19] | 442 infertility cases | XGBoost / RF / SVM / LightGBM | No | Implicit | N/A | AUC up to 0.982 |
| Murugesu et al. [20] | 500 early pregnancies (multi-site) | Elastic-net logistic regression | No | Implicit | N/A (external val.) | AUC 0.91 train / 0.82 external |
| Yang et al. [21] | 1,050 preconception RPL | GBM / RF / GLM / DL | No | Implicit | N/A | Best AUC 0.805 |
| **This work** | **3,000 synthetic patients; 9,098 longitudinal scans; dual-etiology** | **FETA-Transformer + PREG-Net + gated fusion** | **Learned (attention over scans)** | **Explicit (knowledge graph) + cross-attention** | **Yes — input parity + defensibility test** | **CV AUROC 0.884–0.989; fusion ties parameter-free average (see §4)** |

---

## 3. Materials and Methods

### 3.1 Overview of the Proposed Framework

The framework comprises six stages: (1) generation of a synthetic **dual-etiology** first-trimester cohort; (2) multimodal preprocessing with strict patient-level stratified splitting and training-fold-only normalization; (3) on-the-fly derivation of a knowledge-guided patient graph from the same tensors the temporal model consumes; (4) two towers (FETA-Transformer over scan sequences; PREG-Net over the patient graph) and their **gated cross-attention fusion**; (5) a unified training procedure with class-imbalance handling and early stopping on validation AUROC; and (6) evaluation under an input-parity protocol plus attention-based explainability. A single dataset feeds every model family through a shared collation abstraction, so all models consume consistent inputs.

> **Figure 4.** *Proposed framework and gated-fusion architecture* (`figures/fig4_architecture.png`). Left: the two towers (temporal transformer over the scan sequence; knowledge graph over clinical + ultrasound nodes). Centre: the bidirectional masked cross-attention block. Right: the per-patient gate that routes between the two branch logits, plus the correction term, producing the fused logit. Auxiliary per-branch heads (run on the pre-fusion representations) are annotated.

### 3.2 Synthetic Dual-Etiology Cohort Generation

No consented patient data were used. To permit full methodological development while making the absence of real data explicit, a synthetic cohort was generated from biologically informed models under a fixed seed (seed = 42). The generator produces two linked tables — a patient table (maternal covariates and outcome) and a scan table (FHR, CRL, GS, YSD indexed by gestational age) — plus a diagnostics-only channel-label table used solely to verify the cohort's construction (never as a model input).

**Growth curves.** Gestational-age-dependent central tendencies for the four ultrasound channels were specified from published first-trimester references (Hadlock-type CRL, Nyberg-type gestational sac, Doubilet & Benson FHR, Blaas-type yolk sac), each modeled as a smooth function of gestational week modulated by a patient-specific latent growth factor.

**Dual-etiology outcome model — the central design choice.** Each patient's loss label is generated as the **logical OR of two independent channels**:

- a **sonographic channel** (`label_sono`), an embryonic abnormality expressed in the scan trajectory and therefore recoverable from the ultrasound sequence; and
- a **maternal channel** (`label_mat`), a near-deterministic threshold on an interaction of maternal variables, recoverable from the clinical profile,

so that `label = label_sono OR label_mat`. This construction deliberately manufactures **complementary modality signal**: some losses are visible only in the scan, some only in the maternal profile, and some in both. It is the regime in which multimodal fusion is *expected* to help, which is precisely why it is the right stress test for the fusion mechanism. We emphasize two disclosures required by this design:

1. The channel strengths are **specified, not estimated.** The maternal channel in this cohort is made strongly (near-deterministically) predictive by construction via tuned generator constants; it is **not** an empirically measured maternal effect size. Under *published* epidemiological effect sizes the maternal channel would be far weaker, and both the standalone graph model and any fusion would be correspondingly weaker. Any number reported here is a property of the generator, not of real pregnancies.
2. The complementary structure is an **in-silico modelling assumption.** Results demonstrate what an architecture *can* exploit when etiologic heterogeneity is present and strong; they are not evidence about real cohorts.

**Verification of the OR structure.** We confirmed on the full cohort that `label == label_sono OR label_mat` for **100 % of the 3,000 patients**. The channel decomposition of the 564 losses is: **222 sonographic-only, 123 maternal-only, and 219 both-channel** (marginal prevalences: sonographic 14.7 %, maternal 11.4 %). Thus **123 losses (21.8 % of all losses) are invisible to a scan-only model** and **222 (39.4 %) are invisible to a maternal-only model** — a quantified upper bound on how much a single-modality view must miss, and on how much complementary signal a multimodal model can in principle recover.

**Maternal covariates and longitudinal sampling.** Age, BMI, parity, gravidity, previous loss, conception mode (spontaneous/IVF), and singleton status were sampled with realistic correlations, preserving gravidity ≥ parity. Each patient contributes 2–5 scans (mean 3.03, median 3) across gestational weeks 5–14, generated autoregressively so consecutive measurements are correlated; clinically plausible missingness is introduced and resolved by per-patient longitudinal interpolation with a global-median fallback.

### 3.3 Preprocessing, Splitting, and the Input-Parity Rule

Each patient is represented by (i) a temporal tensor of shape (T, 4) over up to T = 5 scans, (ii) a gestational-age vector (T,), (iii) a binary temporal mask (T,), (iv) a maternal-feature vector of length 7, and (v) a binary outcome.

**Patient-level stratified splitting.** All scans from a patient are confined to a single split to prevent within-patient leakage. A stratified 70 % / 15 % / 15 % partition yields **2,099 / 449 / 452 patients** (394 / 84 / 86 losses; test-split loss rate 19.0 %). Continuous features are standardized using statistics computed **exclusively on the training split**; binary maternal features are left unnormalized. Sequences shorter than T = 5 are zero-padded and flagged by the temporal mask.

**The input-parity rule (the protocol's core).** Every model in this study — the two standalone towers, the fusions, and the tabular baselines — receives **both modalities**. In particular, standalone **PREG-Net is given the full multimodal graph** (maternal *and* per-scan ultrasound nodes; §3.4), *not* a maternal-only subgraph. We explicitly do **not** report a "specialist gap" obtained by restricting a tower to one modality: such a gap is guaranteed by the restriction and says nothing about the fusion mechanism. The diagnostics-only channel-label table is never fed to any model (doing so would leak the OR answer); we verified that no training or evaluation path reads it.

### 3.4 Knowledge-Guided Patient Graph

For PREG-Net, each patient is a graph derived on the fly from the same tensors the temporal model uses. Under input parity the **full graph** has **27 nodes** — 20 temporal nodes (four ultrasound channels at each of five scan positions) and 7 static maternal nodes — with node features equal to the (normalized) measurement values and a node mask deactivating padded scans. Edges encode **established physiological relationships** rather than data-inferred correlations; three families give **167 directed edges**:

- **Physiological (40):** intra-scan relationships among ultrasound channels (e.g., FHR↔CRL, YSD↔GS, CRL↔GS, FHR↔YSD), instantiated at each scan position.
- **Temporal (32):** each ultrasound channel connected to itself across consecutive scans (both directions).
- **Maternal → ultrasound (95):** maternal nodes connected to the ultrasound nodes they are believed to influence (age/BMI/prior-loss → all channels; gravidity/parity → FHR/CRL; conception → FHR/CRL/GS), across all scan positions.

For completeness we note a **clinical-only** graph variant (7 maternal nodes, 24 maternal↔maternal edges) exists in the code; it is the handicap the input-parity rule forbids and is **not used** in any reported result.

**Table 2. Knowledge-guided edge families in the full PREG-Net patient graph (input parity).**

| Edge family | Count | Example edges | Physiological rationale |
|---|---|---|---|
| Physiological (intra-scan) | 40 | FHR↔CRL, YSD↔GS, CRL↔GS, FHR↔YSD | Cardiac output influences growth; yolk-sac function supports sac development; growth measures co-vary |
| Temporal (same variable) | 32 | FHR(t)↔FHR(t+1), CRL(t)↔CRL(t+1) | Week-to-week evolution of each biomarker |
| Maternal → ultrasound | 95 | Age→{FHR,CRL,GS,YSD}, BMI→all, Prior-loss→all, Conception→{FHR,CRL,GS} | Maternal status modulates embryonic development and risk |
| **Total** | **167** | — | **27 nodes** (20 temporal + 7 maternal) |

### 3.5 FETA-Transformer (Temporal Tower)

The FETA-Transformer replaces hand-crafted temporal features with learned temporal representations through four mechanisms: **(a)** a **continuous, gestational-age-aware positional encoding**, PE(t, 2i) = sin(t · θ⁻²ⁱ/ᵈ), PE(t, 2i+1) = cos(t · θ⁻²ⁱ/ᵈ), with t the actual gestational age in weeks, so the model perceives the biologically meaningful gap between weeks 6–7 differently from 9–10; **(b)** **modality-specific projections**, each ultrasound channel receiving its own linear map before mixing; **(c)** **attention-based temporal pooling** with a learned query, z = Σₜ αₜ hₜ, αₜ = softmax(wᵀhₜ), padded positions masked, so the weights αₜ are directly interpretable as the gestational weeks relied upon; and **(d)** **maternal cross-attention**, in which the maternal vector acts as a query attending over the pooled temporal representation. A LayerNorm–Linear–GELU–Linear head emits a single logit. The reported configuration uses d_model = 64, four heads, two encoder layers, feed-forward dimension 128, dropout 0.2 (**87,169 trainable parameters**).

A structural caveat that becomes a finding in §4: the maternal pathway is a **late bottleneck.** The scan sequence is pooled to a single vector *first*, and only then do the maternal variables cross-attend to that one token — so maternal information enters after the temporal representation has already been collapsed. This limits how much the maternal channel can shape the prediction, and it is visible in the results.

> **Figure 2.** *Marker trajectories by outcome* (`figures/fig2_marker_trajectories.png`). Mean ± band of FHR and CRL versus gestational week for viable vs loss pregnancies, showing the depressed/declining sonographic-channel trajectories the temporal tower is designed to read.

### 3.6 PREG-Net (Relational Tower)

PREG-Net applies a graph-attention network to the knowledge-guided graph, with message passing implemented by hand in pure PyTorch (scatter-add and sparse softmax; no external graph library). Each layer computes per-edge, per-patient attention coefficients in the standard GAT form, αᵥᵤ = softmax_u(LeakyReLU(aᵀ[Whᵥ ‖ Whᵤ])), so the importance of a relationship varies per patient; node embeddings are updated by attention-weighted aggregation with node masking, and a masked, attention-weighted readout produces a graph embedding that a linear head maps to a logit. It exposes a per-node importance vector and per-layer edge-attention tensors. The reported configuration uses hidden dimension 64, two GAT layers, four heads, dropout 0.2, and 11 node types (**16,130 trainable parameters**, roughly one-fifth of FETA's). Because it treats every clinical and ultrasound variable as a **peer node**, PREG-Net integrates the maternal channel on an equal footing with the ultrasound channels — the architectural difference that explains its strength relative to FETA under input parity.

### 3.7 Gated Hybrid Fusion (the Proposed Architecture)

The **HybridFusionPredictor** performs *intermediate* fusion: the two towers exchange information mid-network, before either commits to a prediction. Its forward pass proceeds as follows.

1. **Pre-fusion encoding.** FETA encodes its per-scan token sequence h_feta ∈ ℝ^{B×T×d} (before pooling); PREG-Net encodes its per-node embeddings h_preg ∈ ℝ^{B×N×d} (before readout). Both towers use d = 64 so the representations are dimension-compatible.
2. **Bidirectional masked cross-attention.** A `CrossModalFusion` block lets FETA tokens attend over PREG-Net nodes and vice versa (multi-head, residual + LayerNorm, padded keys masked), yielding cross-attended representations for each tower.
3. **Complementary branch heads.** The **auxiliary per-branch logits** are produced from the **pure, pre-fusion** representations, and an auxiliary loss keeps each tower individually predictive. Running the branch heads on the pure representations prevents the two towers from collapsing into copies of one another under bidirectional attention — a failure mode that would defeat the purpose of fusion.
4. **Per-patient gated routing.** The cross-attended pooled representations are concatenated, LayerNorm-ed, and passed to two small MLP heads: a **gate** g = σ(·) ∈ (0, 1) and a **correction** c. The fused logit is

   ℓ_fused = g · ℓ_feta + (1 − g) · ℓ_preg + c,

   i.e., a *learned, per-patient* convex routing between the two towers' predictions plus a correction. The gate g is retained as an explainability signal (the per-patient weight the model places on the temporal tower).

The gated hybrid has **153,733 trainable parameters**. Its design intent is to let fusion win precisely on the patients where *exactly one* tower is right — the sonographic-only and maternal-only losses the dual-etiology cohort contains by construction. Whether it realizes that intent, over and above a parameter-free average, is the empirical question of §4.6.

### 3.8 Baselines (all under input parity)

The fusion is judged against baselines that see the **same** two modalities:

- **Ensemble (logit average)** — the *parameter-free* mean of the two towers' logits. This is the decisive comparator: if the gated hybrid cannot beat a trivial average of the same towers, its machinery is not earning its keep.
- **Ensemble (learned)** — two trainable mixing weights over the tower logits.
- **Tabular baselines over engineered per-patient features** (per-channel mean/last/slope, scan count, and the seven maternal features): **HistGradientBoosting, Random Forest, Logistic Regression, and an MLP.** A gradient-boosted tree is the off-the-shelf yardstick a practitioner would reach for first.

### 3.9 Training Procedure and Class-Imbalance Handling

All deep models share one training loop; a collation mode selects the tensors assembled for each family. Because the cohort is imbalanced (18.8 % loss), the objective is weighted binary cross-entropy, `BCEWithLogitsLoss`, with positive-class weight equal to the train-split negative/positive ratio (pos_weight ≈ 4.33); models emit raw logits. Optimization uses AdamW (lr 1×10⁻³, weight decay 1×10⁻⁴), batch size 32, cosine-annealing-with-warm-restarts, with early stopping / checkpointing on **validation AUROC** (patience 15). The two towers train for up to 60 epochs and the gated hybrid for up to 80; the auxiliary per-branch loss weight is 0.3. Training used CPU with seed 42 for reproducibility.

**Table 3. Training configuration and model sizes.**

| Setting | Value |
|---|---|
| Optimizer / LR / weight decay | AdamW / 1×10⁻³ / 1×10⁻⁴ |
| Batch size / schedule | 32 / CosineAnnealingWarmRestarts |
| Early-stopping metric | Validation AUROC (patience 15) |
| Loss | Weighted BCE (pos_weight ≈ 4.33) |
| Auxiliary per-branch weight (hybrid) | 0.3 |
| Device / seed | CPU / 42 |
| FETA-Transformer | d_model 64, 4 heads, 2 layers, d_ff 128, dropout 0.2 — **87,169 params** |
| PREG-Net | hidden 64, 2 GAT layers, 4 heads, dropout 0.2 — **16,130 params** |
| Gated hybrid | cross-attention (4 heads) + per-patient gate + correction — **153,733 params** |

### 3.10 Evaluation Protocol and Metrics

Two complementary evaluations are reported. **(i) Single held-out split:** train on the training split, select on validation, evaluate once on the untouched 452-patient test split. **(ii) Five-fold stratified cross-validation:** the entire pipeline is repeated across five patient-level folds to estimate mean ± SD, which we treat as the more trustworthy estimate. Metrics are AUROC (primary), AUPRC, accuracy, sensitivity, specificity, and F1 at a 0.5 threshold. Because the cohort is imbalanced, we foreground AUROC and AUPRC and note the **no-skill accuracy baseline of 81.2 %** (predicting "viable" for everyone); bare accuracy is never used to rank models.

**The defensibility test.** For the fusion claim specifically, we adopt an explicit rule: the gated hybrid is credited with beating a comparator **only if its mean AUROC gap exceeds the hybrid's own fold-to-fold standard deviation**, and we report the number of folds won. This test — computed, not asserted — is the safeguard against the fusion-paper trap described in §1.2.

---

## 4. Results

### 4.1 Experimental Settings

All experiments used PyTorch (deep models) and scikit-learn (tabular baselines) on CPU with seed 42; single-split and cross-validation protocols shared identical preprocessing, patient-level stratified splitting, and training configuration (Table 3). Deep models were selected on validation AUROC and evaluated on the held-out test split; tabular baselines were fit on engineered per-patient features and evaluated on the same split. Every model saw both modalities (input parity, §3.3).

> **Figure 5.** *Specialist training dynamics* (`figures/fig5_specialist_training.png`): per-epoch training/validation loss and validation AUROC for the two towers, confirming convergence and the early-stopping selection point.

### 4.2 Dataset Characteristics

The synthetic cohort comprised **3,000 patients and 9,098 scans**, with **564 losses (18.8 %)** and 2,436 viable pregnancies. Patients contributed 2–5 scans (mean 3.03, median 3) across gestational weeks 5.0–13.9. Table 4 summarizes the cohort.

**Table 4. Synthetic dual-etiology cohort summary.**

| Attribute | Value |
|---|---|
| Total patients / scans | 3,000 / 9,098 |
| Pregnancy-loss cases | 564 (18.8 %) |
| Viable pregnancies | 2,436 (81.2 %) |
| Scans per patient | 2–5 (mean 3.03, median 3) |
| Gestational-age range | 5.0–13.9 weeks |
| Ultrasound features | FHR, CRL, GS, YSD |
| Maternal features | age, BMI, parity, gravidity, previous loss, conception (IVF/spontaneous), singleton |
| Etiology | dual (`label = label_sono OR label_mat`), verified 100 % of rows |
| Loss decomposition | 222 sonographic-only · 123 maternal-only · 219 both-channel |
| Train / validation / test | 2,099 / 449 / 452 patients (394 / 84 / 86 losses) |
| No-skill accuracy baseline | 81.2 % |
| Random seed | 42 |

> **Figure 1.** *Cohort overview* (`figures/fig1_cohort_overview.png`): outcome balance, scans-per-patient histogram, and the sonographic/maternal/both loss decomposition. **Figure 3.** *Maternal-channel risk structure* (`figures/fig3_maternal_risk.png`): how the maternal interaction thresholds separate `label_mat` cases, illustrating the specified (not estimated) maternal channel strength.

### 4.3 Why Fusion Could Help Here — the Complementary Channel Structure

The dual-etiology construction guarantees genuine complementary signal. Because 123 of 564 losses are maternal-only and 222 are sonographic-only, a perfect scan-only classifier is capped at recovering 342 of 564 losses and a perfect maternal-only classifier at 441 of 564; only a model with access to **both** modalities can reach all of them. This is the sense in which the cohort is a *favorable* setting for multimodal fusion — and therefore a fair place to ask whether the fusion **mechanism** adds anything beyond simply having both modalities available.

### 4.4 Single-Split Model Comparison

On the held-out test split the gated hybrid ranked first, ahead of the parameter-free logit average and the standalone full-graph PREG-Net; the temporal tower trailed (Table 5). Note the complementary operating points: FETA is conservative (specificity 0.989, sensitivity 0.767), while PREG-Net and the fusions recover far more losses.

**Table 5. Held-out test-set performance (single split; 452 patients, 86 losses), ranked by AUROC.**

| Model | Sees | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gated hybrid (fusion)** | both, gated | **0.9975** | 0.9886 | 0.9801 | 0.9767 | 0.9809 | 0.9492 |
| Ensemble (logit average) | both | 0.9944 | 0.9792 | 0.9712 | 0.9186 | 0.9836 | 0.9240 |
| PREG-Net (full multimodal graph) | both | 0.9938 | 0.9782 | 0.9668 | 0.9186 | 0.9781 | 0.9133 |
| FETA-Transformer | both | 0.9141 | 0.8663 | 0.9469 | 0.7674 | 0.9891 | 0.8462 |

Read in isolation, Table 5 looks like a fusion win. It is not — as the cross-validation shows, the margins between the hybrid, the average, and the single graph model are smaller than fold-to-fold noise, and the single-split ordering is not stable. We report Table 5 for completeness and immediately temper it.

> **Figure 6.** *ROC and precision–recall curves on the test split* (`figures/fig6_roc_pr.png`): near-ceiling, tightly clustered curves for every multimodal model, with FETA visibly separated below.

### 4.5 Cross-Validation

Under five-fold stratified cross-validation the models cluster tightly and the single-split ordering collapses (Table 6). The **parameter-free logit average attains the highest mean AUROC**; the gated hybrid is second and within noise of it; a standalone PREG-Net given both modalities reaches 0.981 on its own; and an off-the-shelf gradient-boosted tree ties the hybrid. Only the temporal tower separates from the pack — downward.

**Table 6. Five-fold cross-validation performance (mean ± SD), ranked by mean AUROC. All models under input parity.**

| Model | Sees | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---:|---:|
| Ensemble (logit average) | both | **0.989 ± 0.006** | 0.976 ± 0.012 | 0.926 | 0.985 |
| Ensemble (learned) | both | 0.989 ± 0.004 | 0.974 ± 0.008 | 0.924 | 0.978 |
| **Gated hybrid (fusion)** | both, gated | **0.988 ± 0.004** | 0.972 ± 0.006 | 0.943 | 0.975 |
| HistGradientBoosting | both (raw features) | 0.986 ± 0.004 | — | 0.887 | 0.985 |
| Random Forest | both (raw features) | 0.985 ± 0.004 | — | 0.828 | 0.987 |
| PREG-Net (full multimodal graph) | both | 0.981 ± 0.006 | 0.958 ± 0.013 | 0.902 | 0.976 |
| Logistic Regression | both (raw features) | 0.970 ± 0.006 | — | 0.885 | 0.920 |
| Multilayer Perceptron | both (raw features) | 0.959 ± 0.016 | — | 0.857 | 0.929 |
| FETA-Transformer | both | 0.884 ± 0.014 | 0.844 ± 0.017 | 0.739 | 0.981 |

> **Figure 7.** *Cross-validation AUROC* (`figures/fig7_cv_auroc.png`): point-and-error-bar chart of mean AUROC ± SD, making the overlapping bands among the multimodal models — and FETA's separation — visually explicit.

### 4.6 The Defensibility Test — Does the Gated Fusion Beat Trivial Averaging?

This is the paper's decisive analysis. Applying the rule of §3.10 to the cross-validated results:

| Comparison | Comparator AUROC | Δ (hybrid − comparator) | Folds hybrid wins | Larger than hybrid SD (0.004)? |
|---|---:|---:|:---:|:---:|
| Gated hybrid vs **plain logit average** | 0.989 | **−0.0013** | **1 / 5** | **No** |
| Gated hybrid vs **PREG-Net (both modalities)** | 0.981 | +0.006 | 5 / 5 | Yes (modest) |
| Gated hybrid vs **HistGradientBoosting** | 0.986 | +0.001 | — | No |

The verdict is unambiguous and negative for the architecture: **the gated hybrid's margin over a parameter-free logit average of the same two towers is −0.0013 — it is on average *behind* the trivial average and wins only one of five folds.** The gate, the cross-attention block, and the correction head therefore have a *measured contribution of approximately zero* over simply averaging the two logits. The hybrid does beat a single PREG-Net (+0.006, five of five folds), but that gain is no larger than what a parameter-free average of the same towers already obtains over PREG-Net (+0.008): it reflects the value of *adding the temporal view at all*, not of the gating mechanism. Following our own pre-committed rule, **the honest statement is "ties" against the average, not "beats."**

> **Figure 8.** *Per-fold hybrid − average AUROC* (`figures/fig8_per_fold_delta.png`): the fold-by-fold difference straddling zero. **Figure 9.** *Per-patient gate behavior* (`figures/fig9_gate.png`): although the learned gate varies meaningfully across patients (routing toward FETA for some, PREG-Net for others), that variation does not translate into an aggregate advantage over a fixed 0.5 weighting.

### 4.7 A Real, Unexploited Finding: the Temporal Tower Under-Uses the Maternal Variables

On **identical** inputs, FETA scores 0.884 while PREG-Net scores 0.981 — a ~0.10 AUROC gap that is *not* an artifact of handicapping (both see both modalities). The cause is architectural: FETA's maternal pathway is a **late cross-attention bottleneck** (§3.5). The scan sequence is pooled to a single vector first, and only then do the maternal variables cross-attend to that one token, so the strong maternal channel — near-deterministic in this cohort — cannot fully shape the prediction. PREG-Net, treating all eleven variable types as peer graph nodes, integrates the maternal channel from the first layer. This gap is a legitimate, addressable architecture target: injecting maternal features as sequence tokens, or fusing them earlier, would let the temporal tower use the same signal the graph tower already exploits.

### 4.8 Interpretability

Both towers expose attention-based explanations, exported per patient by the pipeline.

**FETA temporal attention (Figure 10, `figures/fig10_temporal_attention.png`).** Aggregated over the test split, the temporal-pooling weights concentrate on the mid-to-late first-trimester scans, where the sonographic-channel divergence between viable and loss trajectories is largest, while the maternal cross-attention weights fall more heavily on the earliest scans — consistent with maternal context being most informative before strong ultrasound evidence accumulates.

**PREG-Net node and edge attention (Figure 11, `figures/fig11_graph_attention.png`).** Node importance concentrates on fetal heart rate and yolk-sac diameter among the ultrasound nodes and on the maternal nodes that drive the maternal channel; edge attention distributes across physiological growth relationships and same-variable temporal links. As in prior work, the highest-attention edges are not exclusively the specific directed edges designated as the primary design rationale — a partial, honestly reported concordance rather than a clean one.

These views turn both towers from black boxes into inspectable ones; we make no claim that the attention maps constitute clinical validation.

---

## 5. Discussion

**Principal findings.** We set out to test, honestly and under input parity, whether a gated multimodal fusion of a temporal tower and a graph tower beats simple alternatives. Three findings stand out. First, the **multimodal input is highly informative** on this cohort: any model that reads both modalities — the towers' average, the gated hybrid, a single graph network, a gradient-boosted tree — lands at 0.98–0.99 AUROC, and the dual-etiology structure gives fusion genuine complementary signal to recover (123 maternal-only and 222 sonographic-only losses). Second, and decisively, the **gated fusion architecture provides no measurable benefit over a parameter-free logit average of the same two towers** (Δ = −0.0013, one of five folds won); a single PREG-Net given both modalities nearly matches it, and an off-the-shelf tree ties it. Third, on identical inputs the **temporal tower under-uses the maternal variables** because of a late cross-attention bottleneck — a concrete, unexploited gap rather than a property of the data.

**Why the fusion mechanism does not help here.** Two forces combine. (i) *Ceiling.* When both channels are strongly recoverable from their own modality — as they are by construction in this cohort — every multimodal model saturates, and architectural differences cannot be measured at saturation. (ii) *Averaging already captures the complementarity.* The complementary structure (some losses in one channel, some in the other) is exactly what a symmetric combination of two competent towers exploits; a fixed 0.5 average recovers it as well as a learned per-patient gate, because on the patients where one tower is confidently right the average already tips the correct way. The gate's per-patient routing (Figure 9) is real but redundant. The honest reading is that the *value here is in having both modalities*, not in the fusion machinery.

**On the very high scores.** The uniformly high AUROC values are a property of the *generator*, not evidence of a hard, realistic task. The growth curves are smooth and literature-derived, and — critically — the maternal channel's strength is **specified, near-deterministic, and tuned**, not an empirically estimated effect size. Under published epidemiological effect sizes the maternal channel would be far weaker (most early loss is chromosomal → embryonic → sonographic), the standalone graph model would fall well below 0.98, and both fusion and averaging would have less to work with. The numbers must be read as an in-silico demonstration of what an architecture *can* exploit when strong complementary signal is present, not as a claim about real pregnancies.

**Methodological contribution.** The paper's durable contributions are the **input-parity protocol** and the **defensibility test**. Together they close the loophole that makes many fusion results unfalsifiable: by refusing to handicap the single-model baselines and by crediting the fusion only when its gap exceeds its own fold SD, we convert "our fusion beats its sub-models" (true by construction) into a claim that can actually fail — and, here, does. We regard reporting that failure honestly as more useful than a ceiling-inflated win.

**Practical implication.** On data like this, a practitioner should prefer a parameter-free average of the two towers, or a single graph model, over the gated hybrid, on grounds of parsimony and stability. The gated architecture would need a **harder, non-saturated** regime — or real, noisy data — to demonstrate value, and the temporal tower's maternal bottleneck should be fixed before any fusion comparison is repeated.

---

## 6. Limitations

1. **The data are entirely synthetic.** No real patients are involved. All reported performance measures how well the models recover structure the generator built in; it is not clinical evidence.
2. **The cohort is dual-etiology by design, with specified channel strengths.** `label = label_sono OR label_mat` is a modelling assumption, and the maternal channel is made strong by tuned constants, not estimated from data. Results are an in-silico demonstration, not a measured property of pregnancy.
3. **The gated fusion does not beat simple baselines.** Under cross-validation it ties a parameter-free logit average, a single multimodal PREG-Net, and a gradient-boosted tree. On this cohort, architectural sophistication in the fusion head is not justified by performance.
4. **Ceiling effects.** Near-saturated metrics mean decimal-level differences are not statistically resolvable; architecture cannot be discriminated at this separability. Testing the gate meaningfully requires a deliberately harder cohort.
5. **The temporal tower is not at its architectural best.** FETA's maternal-pathway bottleneck depresses its standalone score and, by extension, the towers it feeds; the fusion comparison should be revisited after that gap is closed.
6. **No external validation, calibration, or decision-curve analysis**, and **no component ablation** of the gate/cross-attention/correction beyond the average comparison. These are prerequisites for any clinical or architectural claim.
7. **Fixed knowledge graph and limited temporal depth.** The graph is hand-specified and static across patients (only its attention weights vary), and sequences are truncated to five scans.

---

## 7. Conclusion

We presented a gated multimodal fusion architecture for first-trimester pregnancy-loss prediction — a temporal-attention transformer and a knowledge-guided graph-attention network, exchanging information through cross-attention and combined by a learned per-patient gate — and, more importantly, an **input-parity evaluation protocol with an explicit defensibility test** designed to determine honestly whether such fusion is worth its complexity. On a documented synthetic dual-etiology cohort of 3,000 patients, engineered to contain strong complementary sonographic and maternal signal, we find that the multimodal input is highly informative but that the **gated fusion mechanism provides no measurable advantage over a parameter-free logit average of the same two towers** (five-fold AUROC 0.988 ± 0.004 vs 0.989 ± 0.006; the hybrid wins one of five folds), while a single graph model given both modalities reaches 0.981 and a gradient-boosted tree ties the fusion. We additionally identify a concrete, addressable architectural gap: the temporal tower under-uses the maternal variables because of a late cross-attention bottleneck. We therefore frame this work as a **methodological, reproducibility, and evaluation-protocol contribution** — a documented, interpretable, end-to-end pipeline and a candid, parity-controlled appraisal of where multimodal fusion does and does not help. The essential next steps are: repeating the comparison on a deliberately harder (non-saturated) cohort and on real, consented, multi-center longitudinal data; fixing the temporal tower's maternal pathway; and adding formal calibration, decision-curve analysis, and component ablations. Only after those steps could any claim of clinical or architectural utility be responsibly advanced.

---

## Data and Code Availability

All data used in this study are synthetic and are regenerable deterministically from the released generator under a fixed seed (seed = 42); no real patient data were used or distributed. The dual-etiology synthetic-cohort generator, the model implementations (FETA-Transformer, PREG-Net, and the gated HybridFusionPredictor, in pure PyTorch), the training and cross-validation scripts, the input-parity comparison and defensibility-test code, and the notebook that reproduces the full workflow end-to-end are available in the project repository. Derived result artifacts (single-split and cross-validation metric tables and the exported figures) are included.

## Ethics Statement

This study did not involve human participants, human tissue, or identifiable personal data. All analyses were performed on computationally generated synthetic data; accordingly, institutional review board approval and informed consent were not applicable. No results reported here constitute clinical evidence, and prospective ethical review, consent procedures, and regulatory assessment would be required before any use on real patient data.

## Conflict of Interest

The authors declare no conflict of interest.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## References

Note: References [15]–[21] correspond to the comparator studies discussed in Section 2 and Table 1; [24]–[25] are the core architectural references.

[1] Mikołaj KW, Christensen AN, Taksøe-Vester CA, et al. Predicting abnormal fetal growth using deep learning. *NPJ Digital Medicine*. 2025;8(1):318.

[2] Bhavani G, Jeyalakshmi C. Prediction of clinical risk factors in pregnancy using optimized neural network scheme. *Placenta*. 2025;163:33–42.

[3] Pavanya M, Chadaga K, Vasudeva A, Rao BK, Prabhu S, Bhat SK. Prediction of birthweight with early and mid-pregnancy antenatal markers utilising machine learning and explainable artificial intelligence. *Scientific Reports*. 2025;15(1):26223.

[4] Ferreira I, Simões J, Pereira B, Correia J, Areia AL. Ensemble learning for fetal ultrasound and maternal–fetal data to predict mode of delivery after labor induction. *Scientific Reports*. 2024;14(1):15275.

[5] Mennickent D, Rodríguez A, Opazo MC, et al. Machine learning applied in maternal and fetal health: a narrative review focused on pregnancy diseases and complications. *Frontiers in Endocrinology*. 2023;14:1130139.

[6] Hua Q, Yang F, Zhou Y, Shi F, You X, Guo J, Li L. Predictive models using machine learning to identify fetal growth restriction in patients with preeclampsia. *Journal of Medical Internet Research*. 2025;27:e70068.

[7] Bertini A, Salas R, Chabert S, Sobrevia L, Pardo F. Using machine learning to predict complications in pregnancy: a systematic review. *Frontiers in Bioengineering and Biotechnology*. 2022;9:780389.

[8] Wang C, Johansson ALV, Nyberg C, et al. Prediction of pregnancy-related complications in women undergoing assisted reproduction, using machine learning methods. *Fertility and Sterility*. 2024;122(1):95–105.

[9] Zhu Z, Wei N, Guo J, et al. Predicting the risk of threatened abortion using machine learning methods: a comparative study. *BMC Pregnancy and Childbirth*. 2025;25(1):901.

[10] Magnus MC, Wilcox AJ, Morken NH, Weinberg CR, Håberg SE. Role of maternal age and pregnancy history in risk of miscarriage: prospective register based study. *BMJ*. 2019;364:l869.

[11] Cavalcante MB, Sarno M, Peixoto AB, Araujo Júnior E, Barini R. Obesity and recurrent miscarriage: a systematic review and meta-analysis. *Journal of Obstetrics and Gynaecology Research*. 2019;45(1):30–38.

[12] Wu S, Dong J, Shi J, et al. Machine learning prediction of short cervix in mid-pregnancy based on multimodal data from the first-trimester screening period. *Biomedicines*. 2025;13(9):2057.

[13] Zhang M, Sheng J. The application of super-resolution ultrasound radiomics models in predicting the failure of conservative treatment for ectopic pregnancy. *Reproductive Biology and Endocrinology*. 2025;23(1):102.

[14] Lee LH, Bradburn E, Craik R, et al. Machine learning for accurate estimation of fetal gestational age based on ultrasound images. *NPJ Digital Medicine*. 2023;6(1):36.

[15] Liu L, Zang Y, Zheng H, et al. An AI method to predict pregnancy loss by extracting biological indicators from embryo ultrasound recordings in early pregnancy. *Scientific Reports*. 2025;15(1):25946.

[16] Liu C, Wei X, Wang F. The predictive value of ultrasound markers for pregnancy outcomes in recurrent pregnancy loss: a retrospective study. *Scientific Reports*. 2024;14(1):16657.

[17] Yan S, Xiong F, Xin Y, Zhou Z, Liu W. Optimizing evaluation of endometrial receptivity in recurrent pregnancy loss: a preliminary investigation integrating radiomics from multimodal ultrasound via machine learning. *Frontiers in Endocrinology*. 2024;15:1380829.

[18] Yan S, Xiong F, Xin Y, Zhou Z, Liu W. Automated assessment of endometrial receptivity for screening recurrent pregnancy loss risk using deep learning-enhanced ultrasound and clinical data. *Frontiers in Physiology*. 2024;15:1404418.

[19] Zhang J, Liu S, Rong Y, et al. Prediction of uterine cavity conception environment using two-dimensional transvaginal ultrasound imaging semantic feature-based machine learning: a case-control study. *BMC Pregnancy and Childbirth*. 2025;25(1):1204.

[20] Murugesu S, Linton-Reid K, Barcroft J, et al. Reducing uncertainty in early pregnancy: using clinical features and radiomics to develop a machine learning model to predict outcome. *Human Reproduction*. 2025;40(Suppl 1):deaf097-219.

[21] Yang X, Wang R, Zhang W, Yang Y, Wang F. Predicting risk of the subsequent early pregnancy loss in women with recurrent pregnancy loss based on preconception data. *BMC Women's Health*. 2024;24(1):381.

[22] Doubilet PM, Benson CB. Embryonic heart rate in the early first trimester: what rate is normal? *Journal of Ultrasound in Medicine*. 1995;14(6):431–434.

[23] Hadlock FP, Shah YP, Kanon DJ, Lindsey JV. Fetal crown–rump length: reevaluation of relation to menstrual age with high-resolution real-time US. *Radiology*. 1992;182(2):501–505.

[24] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in Neural Information Processing Systems*. 2017;30:5998–6008.

[25] Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. *International Conference on Learning Representations (ICLR)*. 2018.

---

*Manuscript prepared as a methodological, reproducibility, and evaluation-protocol study. All quantitative results were generated from the accompanying synthetic dual-etiology pipeline under an input-parity protocol; the figures referenced above are the plots exported by the reproduction notebook (`figures/fig1`–`fig11`).*
