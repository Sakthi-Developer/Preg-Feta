# Temporal-Attention and Knowledge-Guided Graph Learning for Explainable First-Trimester Pregnancy-Loss Prediction: A Synthetic-Cohort Study of the FETA-Transformer and PREG-Net

**Mr. A. V. M. Kumaran¹\*, Dr. M. Sundara Rajan²**

¹Research Scholar, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035. Email: avmprofessor@gmail.com
²Associate Professor, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035.

*\*Corresponding author:* Mr. A. V. M. Kumaran, Research Scholar, PG & Research Department of Computer Science, Government Arts College (Autonomous), Nandanam, Chennai – 600 035. Email: avmprofessor@gmail.com

---

## Abstract

Early pregnancy loss affects approximately one in five clinically recognized pregnancies and remains one of the most common complications of the first trimester. Fetal heart rate and early ultrasound growth parameters are established markers of embryonic viability, yet their clinical interpretation still relies heavily on fixed thresholds, isolated single-marker readings, and subjective assessment. Two structural weaknesses recur across the predictive-modeling literature: longitudinal ultrasound information is collapsed into hand-crafted summary statistics that discard the temporal shape of embryonic development, and the biological interdependence among cardiac, growth, and maternal factors is treated as a set of independent predictors rather than a coupled system. This study proposes and evaluates two complementary deep-learning architectures that target these weaknesses directly. The **FETA-Transformer** (Fetal Echocardiographic Temporal Attention Transformer) models each patient's variable-length sequence of scans with a continuous, gestational-age-aware positional encoding, modality-specific input projections, attention-based temporal pooling, and a maternal-feature cross-attention head, replacing engineered temporal features with learned temporal representations. The **PREG-Net** (Pregnancy Risk Evaluation Graph Network) casts the same patient as a knowledge-guided graph — 27 nodes spanning per-scan ultrasound measurements and maternal variables, connected by 167 physiologically motivated edges — and applies a hand-implemented graph-attention network that yields node- and edge-level attributions. A late-fusion **ensemble** combines the temporal and relational views. Because no consented patient data were available, all experiments were conducted on a fully synthetic cohort of 800 patients (2,408 scans; 23.5 % loss rate) generated from literature-derived growth curves, with strict patient-level stratified splitting and training-fold-only normalization. On a held-out test split the ensemble reached the highest area under the receiver operating characteristic curve (AUROC 0.9985), followed by the FETA-Transformer (0.9941) and PREG-Net (0.9885); however, under five-fold cross-validation all deep and tabular models clustered within overlapping standard-deviation bands (AUROC 0.976–0.993), and a regularized logistic-regression baseline was statistically indistinguishable from the proposed models. Attention analysis produced clinically legible explanations: FETA concentrated temporal weight on mid-to-late first-trimester scans, and PREG-Net ranked fetal heart rate and yolk-sac diameter as the dominant nodes. We interpret these findings cautiously: they demonstrate that the two architectures are implementable, interpretable, and behave as designed, but — because the cohort is synthetic and simple baselines are highly competitive — they constitute a methodological and reproducibility contribution rather than clinical evidence.

**Keywords:** Early pregnancy loss, fetal heart rate, longitudinal ultrasound, transformer attention, graph neural networks, explainable deep learning, synthetic clinical data

---

## 1. Introduction

### 1.1 Background

The first trimester is a biologically decisive interval during which the majority of clinically recognized pregnancy losses occur. Spontaneous loss before completion of the first trimester affects an estimated 10–20 % of recognized pregnancies and imposes substantial emotional, psychological, and clinical burden on affected individuals and on health systems. Transvaginal and transabdominal ultrasound provide the principal non-invasive window onto early embryonic development, and several quantitative markers acquired during routine first-trimester scanning carry prognostic value. Fetal heart rate (FHR) reflects early cardiac function and is a sensitive index of embryonic well-being; crown–rump length (CRL) is the standard measure of embryonic growth and gestational dating; gestational-sac (GS) dimensions and yolk-sac diameter (YSD) characterize the developing intrauterine and peri-embryonic support structures. Maternal factors — age, body-mass index, obstetric history, mode of conception, and parity — modulate the baseline risk against which these ultrasound signals are read.

Advances in imaging, electronic health records, and computational modeling have made increasingly rich early-pregnancy data available, and machine learning offers a route to integrate these heterogeneous signals into calibrated, individualized risk estimates. In obstetric research, such methods increasingly support risk stratification and clinical decision support. Their application to early pregnancy-loss prediction, however, remains uneven, and several methodological gaps limit clinical translation.

### 1.2 Challenges and Motivation

Four recurring limitations motivate this work.

First, **static and threshold-based interpretation.** Much of clinical practice and a large fraction of the modeling literature reduce FHR to a fixed cut-off (for example, below 100 beats per minute at a given gestational age). Such thresholds are simple but ignore the physiological reality that embryonic cardiac activity rises steeply and non-linearly across the early weeks; a single value stripped of its gestational context and its trajectory is an impoverished predictor.

Second, **single-marker reductionism.** Many studies model one predictor — most often FHR — in isolation, discarding the complementary information carried by growth parameters and maternal variables and, more importantly, the *interactions* among them. Early gestation is a tightly coupled system in which cardiac, growth, placental, and maternal processes co-evolve; models that treat features as independent cannot represent this coupling.

Third, **loss of longitudinal structure.** Where multiple scans are available, they are frequently summarized into scalar features (mean, slope, last value). This hand-crafted aggregation imposes the analyst's prior about which temporal patterns matter and discards the ordering and spacing of observations — precisely the information needed to detect the subtle week-to-week deviations that can precede overt loss.

Fourth, **opacity.** When flexible machine-learning models are used, they often operate as black boxes that expose neither which variables drove a prediction nor which physiological relationships were implicated. This lack of transparency erodes clinical trust and impedes adoption in obstetric workflows where a risk estimate must be explainable to be actionable.

### 1.3 Problem Statement

Taken together, current approaches to early pregnancy-loss prediction exhibit limited ability to (i) represent the *temporal trajectory* of ultrasound markers without manual feature engineering, (ii) represent the *biological interdependence* of cardiac, growth, and maternal factors explicitly rather than implicitly, and (iii) produce explanations that map onto clinical reasoning. A predictive framework is needed that learns temporal and relational structure directly from longitudinal, multimodal data while remaining interpretable and while adhering to rigorous leakage control.

### 1.4 Research Gap

A gap exists at the intersection of longitudinal temporal modeling, explicit relational modeling, and explainability for first-trimester loss prediction. Transformer architectures have transformed sequence modeling but have rarely been applied to the short, irregularly spaced, multimodal sequences typical of early-pregnancy ultrasound. Graph neural networks can encode domain knowledge as structure, but knowledge-guided graphs built from established physiological relationships — as opposed to graphs inferred from data, which risk encoding spurious correlations — have not been developed for this problem. Crucially, no prior work has systematically compared a temporal-attention model and a knowledge-guided relational model on a common cohort under a shared evaluation protocol, nor combined them to test whether the two views are complementary.

### 1.5 Objectives

The primary objective of this study is to design, implement, and rigorously evaluate two complementary, interpretable deep-learning architectures for first-trimester pregnancy-loss prediction and to characterize their behavior against strong tabular baselines. Secondary objectives are to: (a) model irregular longitudinal ultrasound sequences without hand-crafted temporal features; (b) encode physiological relationships explicitly through a knowledge-guided graph; (c) evaluate whether late fusion of the temporal and relational views yields complementary gains; (d) extract clinically legible attention-based explanations from both models; and (e) do all of the above under a transparent, reproducible protocol with strict patient-level splitting and leakage control. Because consented patient data were unavailable, an explicit further objective is to construct a biologically plausible synthetic cohort that permits full methodological development while making the absence of real-world validation unambiguous.

### 1.6 Contributions

This study makes the following contributions:

1. **FETA-Transformer** — a temporal-attention Transformer for longitudinal first-trimester ultrasound that handles irregular scan timing through a continuous, gestational-age-based positional encoding, learns modality-specific projections for each ultrasound channel, aggregates scans by attention-based pooling, and fuses maternal covariates through cross-attention. To our knowledge this is the first application of this design to early pregnancy-loss prediction.

2. **PREG-Net** — a knowledge-guided graph-attention network in which nodes are per-scan ultrasound measurements and maternal variables and edges encode established physiological relationships; the network is implemented in pure PyTorch (hand-written sparse message passing, no external graph libraries) and produces node- and edge-level attributions.

3. **A unified, reproducible evaluation** of both models, a late-fusion ensemble (average and learned), and five tabular baselines on a common synthetic cohort, reporting both single-split and five-fold cross-validated metrics with explicit uncertainty.

4. **An explainability analysis** that turns the models' attention weights into clinically interpretable temporal profiles, node-importance rankings, and edge-attention comparisons against the encoded physiological rationale, together with representative per-patient case studies.

5. **A candid methodological appraisal** — including a fully documented synthetic data generator and an honest account of where simple baselines match the proposed models — intended to support reproducibility rather than to overstate clinical readiness.

---

## 2. Related Work

Recent research on early-pregnancy outcome prediction increasingly integrates ultrasound imaging, structured clinical markers, and machine learning. We summarize representative studies and position the present work against them; Table 1 provides a structured comparison.

**AI-driven biometric extraction and ensemble prediction.** Liu et al. developed a two-stage framework in which a modified U-Net (A3F-net) automatically measures gestational-sac area, YSD, CRL, and FHR from early-pregnancy ultrasound video, feeding these biometrics into an ensemble learner; a CatBoost model reached an AUC of 0.969 for loss prediction, with high biometric-extraction precision and close agreement with physician measurements. This line of work demonstrates that automated measurement can remove operator variability, but it treats the extracted biometrics as static tabular inputs and does not model their temporal trajectory or mutual dependence.

**Week-specific risk modeling.** In a recurrent-pregnancy-loss cohort, Liu et al. built week-wise models combining ultrasound indices, demographics, and serum markers, showing that the informative predictors change with gestational week (age and progesterone early, CRL from week six) and that discrimination improves as pregnancy advances (AUC 0.671 → 0.872 from week five to seven). This underscores the value of temporal, week-aware modeling — but the models remain separate per week rather than integrating the trajectory within a single learner.

**Radiomics and multimodal fusion.** Yan et al. extracted radiomic features from multimodal transvaginal ultrasound to assess endometrial receptivity in unexplained recurrent loss, with an XGBoost combined model reaching AUC 0.871/0.844 (train/test) and SHAP highlighting elastography and clinical features; a follow-up fused a ResNet-50 image model with TabNet for tabular data (fusion AUC 0.853). Zhang et al. used structured semantic features from two-dimensional ultrasound to characterize the uterine cavity environment (test AUC up to 0.982). These works confirm that multimodal fusion and explainability improve performance and trust, but they rely on image-derived features and do not model longitudinal biometrics as sequences or as an explicit relational graph.

**Clinical-radiomic fusion and preconception models.** Murugesu et al. combined radiomic and clinical features via elastic-net logistic regression across two hospitals (train AUC 0.91; external AUC 0.82; recall 0.81), providing valuable external validation. Yang et al. built a preconception risk model for subsequent early loss in recurrent-loss patients (best AUC 0.805 train, 0.781 validation). Related efforts address gestational-age estimation from images (Lee et al., mean absolute error 3.0–4.3 days) and prediction of miscarriage-management success (Murugesu et al., AUC 0.63–0.72). These studies establish the clinical framing and the importance of calibration, external validation, and decision-curve analysis.

**Positioning of the present work.** The proposed framework differs from all of the above in two respects. First, rather than summarizing longitudinal ultrasound into scalar features, the FETA-Transformer *learns* temporal representations directly from irregularly spaced sequences. Second, rather than modeling feature interactions implicitly (through tree splits or SHAP interaction terms), PREG-Net encodes physiological relationships *explicitly* as a knowledge-guided graph and exposes node- and edge-level reasoning. The study also departs from the comparators in an important methodological way: it is conducted entirely on synthetic data and is framed as a reproducibility and methods contribution, with strong tabular baselines reported honestly alongside the proposed models.

**Table 1. Comparative summary of related work and the present study.**

| Ref. | Data / setting | Method | Temporal modeling | Interaction modeling | Explainability | Reported performance |
|---|---|---|---|---|---|---|
| Liu et al. [15] | 630 ultrasound videos (6–10 wk) | A3F-net (U-Net) + ensemble (CatBoost) | No (static biometrics) | Implicit (tree ensemble) | Feature importance | AUC 0.969 |
| Liu et al. [16] | RPL cohort; ultrasound + serum | Week-wise logistic regression | Per-week (separate models) | None | Risk factors per week | AUC 0.671–0.872 |
| Yan et al. [17] | 346 RPL vs 369 controls; multimodal TVUS | XGBoost + radiomics + SHAP | No | Implicit | SHAP | AUC 0.871 / 0.844 |
| Yan et al. [18] | Ultrasound + clinical | ResNet-50 + TabNet fusion | No | Fusion | Nomogram | Fusion AUC 0.853 |
| Zhang et al. [19] | 442 infertility cases | XGBoost / RF / SVM / LightGBM | No | Implicit | Semantic features | AUC up to 0.982 |
| Murugesu et al. [20] | 500 early pregnancies (multi-site) | Elastic-net logistic regression | No | Implicit | Coefficients | AUC 0.91 train / 0.82 external |
| Yang et al. [21] | 1,050 preconception RPL | GBM / RF / GLM / DL | No | Implicit | DCA / calibration | Best AUC 0.805 |
| **This work** | **800 synthetic patients; 2,408 longitudinal scans** | **FETA-Transformer + PREG-Net + ensemble** | **Learned (attention over scans)** | **Explicit (knowledge graph) + cross-attention** | **Temporal attention + node/edge attribution** | **Test AUROC 0.9885–0.9985; CV AUROC 0.976–0.987 (see §4)** |

---

## 3. Materials and Methods

### 3.1 Overview of the Proposed Framework

The framework comprises six stages: (1) construction of a synthetic first-trimester cohort; (2) multimodal preprocessing with strict patient-level splitting and training-fold-only normalization; (3) derivation, on the fly, of a knowledge-guided patient graph from the same tensors used by the temporal model; (4) three model families — the FETA-Transformer over scan sequences, PREG-Net over the patient graph, and a late-fusion ensemble; (5) a unified training procedure with class-imbalance handling and early stopping on validation AUROC; and (6) evaluation and attention-based explainability. A single dataset feeds all three model families through a shared collation abstraction, so the temporal model, the graph model, and the ensemble all consume consistent inputs.

> **Figure 1 (placeholder).** *Proposed framework overview.* A left-to-right block diagram with six labeled stages: **(a)** Synthetic cohort generation (literature growth curves → patients.csv, scans.csv); **(b)** Preprocessing (stratified patient-level split; train-only z-score normalization; padding/masking to T = 5 scans); **(c)** Two parallel representations — a scan-sequence tensor feeding FETA-Transformer, and an on-the-fly knowledge graph feeding PREG-Net; **(d)** Late-fusion ensemble (average / learned weights); **(e)** Weighted BCE training with early stopping on validation AUROC; **(f)** Evaluation + attention explainability. Arrows should show that the graph is *derived* from the same temporal/maternal tensors rather than from a separate data source.

### 3.2 Synthetic Cohort Generation

No consented patient data were used. To permit full methodological development while making the absence of real data explicit, a synthetic cohort was generated from biologically informed models rather than sampled from any real dataset. The generator is deterministic under a fixed seed (seed = 42) and produces two linked tables: a patient table (one row per patient: maternal covariates and outcome label) and a scan table (one row per scan: FHR, CRL, GS, YSD indexed by gestational age).

**Growth curves.** Gestational-age-dependent central tendencies for the four ultrasound channels were specified from published first-trimester references (Hadlock-type CRL growth, Nyberg-type gestational-sac growth, Doubilet & Benson FHR trajectories, and Blaas-type yolk-sac growth). Each channel is modeled as a smooth function of gestational week modulated by a patient-specific latent growth factor.

**Latent biological factors and outcome.** For each patient, latent factors govern overall growth vigor, cardiac health, and placental/yolk-sac support; loss cases are drawn to exhibit systematically depressed and more variable trajectories, consistent with the clinical observation that impaired growth and low or declining FHR precede loss. The overall loss rate was targeted at approximately one in four to reflect a realistically imbalanced cohort.

**Maternal covariates.** Age, BMI, parity, gravidity, previous loss, conception mode (spontaneous/IVF), and singleton status were sampled with realistic correlations (for example, higher maternal age and BMI, higher prior-loss probability, and greater IVF use in the loss group), preserving the biological constraint gravidity ≥ parity.

**Longitudinal sampling and missingness.** Each patient contributes between two and five scans at plausible gestational weeks; scans are generated autoregressively so that consecutive measurements within a patient are correlated. Clinically plausible missingness is introduced and then resolved by per-patient longitudinal interpolation with a global-median fallback, mirroring routine data cleaning.

The resulting cohort characteristics are reported in Table 5 (§4.2). We emphasize that these data are *simulated*: the growth curves encode population-level priors, and the generator does not draw from, and is not validated against, any real patient records.

### 3.3 Multimodal Data Representation and Preprocessing

Each patient is represented by (i) a temporal tensor of shape (T, 4) holding the four ultrasound channels across up to T = 5 scans, (ii) a vector of gestational ages (T,), (iii) a binary temporal mask (T,) marking valid versus padded scan positions, (iv) a maternal-feature vector of length 7, and (v) a binary outcome label.

**Patient-level stratified splitting.** All scans from a given patient are confined to a single split to prevent leakage of within-patient information across training and evaluation. A stratified partition preserves the outcome ratio across subsets, yielding a 70 % / 15 % / 15 % train / validation / test division (559 / 119 / 122 patients; the test split contains 29 loss and 93 viable pregnancies).

**Training-fold-only normalization.** Continuous features are standardized to zero mean and unit variance using statistics computed *exclusively on the training split*; the fixed statistics are then applied to validation and test data. Binary maternal features (previous loss, IVF conception, singleton) are deliberately left unnormalized. This protocol removes a common source of optimistic bias.

**Padding and masking.** Sequences shorter than T = 5 are zero-padded and flagged by the temporal mask, so that attention and pooling operations ignore padded positions.

### 3.4 Knowledge-Guided Patient Graph

For PREG-Net, each patient is represented as a graph derived on the fly from the same temporal and maternal tensors. The graph has **27 nodes**: 20 temporal nodes (the four ultrasound channels at each of five scan positions) and 7 static maternal nodes. Node features are the (normalized) measurement values; a node mask deactivates nodes corresponding to padded scans.

Edges encode **established physiological relationships** rather than data-inferred correlations, which keeps the structure biologically interpretable and avoids encoding spurious associations. Three edge families give **167 directed edges** in total:

- **Physiological edges (40):** intra-scan relationships among ultrasound channels motivated by embryology — for example FHR↔CRL (cardiac output and nutrient delivery influence growth), YSD↔GS (yolk-sac function supports sac development), and CRL↔GS (embryonic and sac growth co-vary), instantiated at each of the five scan positions.
- **Temporal edges (32):** each ultrasound channel connected to itself across consecutive scans (both directions), representing the evolution of a variable over time.
- **Maternal→ultrasound edges (95):** maternal nodes connected to the ultrasound nodes they are believed to influence — age and BMI and prior loss to all channels, gravidity and parity to FHR/CRL, and conception mode to FHR/CRL/GS — across all scan positions.

The edge set and its physiological rationale are summarized in Table 2.

**Table 2. Knowledge-guided edge families in the PREG-Net patient graph.**

| Edge family | Count | Example edges | Physiological rationale |
|---|---|---|---|
| Physiological (intra-scan) | 40 | FHR↔CRL, YSD↔GS, CRL↔GS, FHR↔YSD | Cardiac output influences growth; yolk-sac function supports sac development; growth measures co-vary |
| Temporal (same variable) | 32 | FHR(t)→FHR(t+1), CRL(t)→CRL(t+1) | Week-to-week evolution of each biomarker |
| Maternal → ultrasound | 95 | Age→{FHR,CRL,GS,YSD}, BMI→all, Prior-loss→all, Conception→{FHR,CRL,GS} | Maternal status modulates embryonic development and risk |
| **Total** | **167** | — | 27 nodes (20 temporal + 7 maternal) |

### 3.5 FETA-Transformer

The FETA-Transformer replaces hand-crafted temporal features with learned temporal representations. Its forward pass has four innovations over a vanilla Transformer encoder.

**(a) Continuous, gestational-age-aware positional encoding.** Standard Transformers assume regularly spaced positions. Because ultrasound scans are irregularly spaced, positions are encoded from the *actual gestational age* t (in weeks) rather than the sequence index:

$$\mathrm{PE}(t, 2i) = \sin\!\left(t\,\theta^{2i/d}\right), \qquad \mathrm{PE}(t, 2i+1) = \cos\!\left(t\,\theta^{2i/d}\right),$$

so that the model perceives the biologically meaningful gap between, say, weeks 6 and 7 differently from weeks 9 and 10.

**(b) Modality-specific projections.** Each ultrasound channel m receives its own learnable linear projection $h_{m,t} = W_m x_{m,t} + b_m$ before mixing, allowing channel-specific representations prior to self-attention.

**(c) Attention-based temporal pooling.** Rather than a [CLS] token or mean pooling, scans are aggregated with a learned query w:

$$z = \sum_{t=1}^{T} \alpha_t\, h_t, \qquad \alpha_t = \frac{\exp(w^\top h_t)}{\sum_{t'} \exp(w^\top h_{t'})},$$

with padded positions masked out. The weights $\alpha_t$ are directly interpretable as the gestational weeks the model relied upon.

**(d) Maternal cross-attention.** Static maternal features are not concatenated; they act as a query attending over the temporal representation, $z_{\text{final}} = \mathrm{CrossAttn}(Q = W_m m,\, K = W_k z,\, V = W_v z)$, so that maternal characteristics dynamically re-weight aspects of the trajectory. A final linear classifier produces a single logit.

In the reported configuration the encoder uses model dimension 64, four attention heads, two layers, feed-forward dimension 128, and dropout 0.2 (**87,169 trainable parameters**). The model returns the logit together with an attention dictionary (temporal-pooling weights and maternal cross-attention weights) that downstream explainability code consumes.

> **Figure 2 (placeholder).** *FETA-Transformer architecture.* A vertical schematic: input (T×4 scan tensor + gestational ages + mask) → modality-specific projections → additive continuous positional encoding → LayerNorm/dropout → 2-layer Transformer encoder (4 heads) → attention pooling (show the learned query and α_t weights) → maternal cross-attention block (maternal vector as query attending to pooled representation) → linear classifier → logit. Annotate the two attention outputs that feed the explainability module.

### 3.6 PREG-Net

PREG-Net applies a graph-attention network to the knowledge-guided patient graph. Message passing is implemented by hand in pure PyTorch using scatter-add and sparse softmax — no external graph library is used — which keeps the attention mechanics fully transparent and portable.

Each graph-attention layer computes per-edge, per-patient attention coefficients in the style of a graph attention network:

$$\alpha_{vu} = \frac{\exp\!\big(\mathrm{LeakyReLU}(a^\top [W h_v \,\|\, W h_u])\big)}{\sum_{k \in \mathcal{N}(v)} \exp\!\big(\mathrm{LeakyReLU}(a^\top [W h_v \,\|\, W h_k])\big)},$$

so that the importance of a relationship (for example FHR↔CRL versus YSD↔GS) varies per patient, reflecting different failure mechanisms. Node embeddings are updated by attention-weighted aggregation over neighbors, with node masking so that padded scans contribute nothing. After the graph layers, a masked readout produces a graph-level embedding that a linear head maps to a logit. The network exposes two explainability signals: a per-node importance vector and the per-layer, per-head edge-attention tensors.

The reported configuration uses hidden dimension 64, two graph-attention layers, four heads, and dropout 0.2 (**16,130 trainable parameters** — roughly one-fifth of FETA's), consistent with the low computational cost expected of a graph model with fewer than thirty nodes.

> **Figure 3 (placeholder).** *PREG-Net architecture and knowledge graph.* Two panels. **Left:** the 27-node knowledge graph laid out with the 20 temporal ultrasound nodes (grouped by scan position) and 7 maternal nodes, edges colored by family (physiological, temporal, maternal→ultrasound). **Right:** the processing pipeline — node features → 2 stacked graph-attention layers (4 heads, show per-edge α_{vu}) → masked graph readout → linear classifier → logit — with callouts for the node-importance and edge-attention outputs.

### 3.7 Ensemble (Late Fusion)

The ensemble runs both sub-models and fuses their logits. Two variants are evaluated: an **average** fusion (equal weights) and a **learned** fusion in which the two mixing weights are trainable parameters. The learned variant converged to weights of approximately 0.542 (FETA) and 0.481 (PREG-Net), indicating that both views contribute and that neither is redundant. The ensemble has **103,299 trainable parameters** in total and returns a fused logit alongside the individual sub-model logits and their attention dictionaries.

### 3.8 Training Procedure and Class-Imbalance Handling

All deep models are trained with a single unified procedure to ensure comparability. A collation mode selects which tensors are assembled for each family: temporal-plus-maternal tensors for FETA, the derived graph for PREG-Net, and both for the ensemble.

**Loss and imbalance.** Because the cohort is imbalanced (≈ 23.5 % loss), the objective is a weighted binary cross-entropy, `BCEWithLogitsLoss`, with a positive-class weight equal to the train-split negative/positive ratio ($w = N_{\text{neg}}/N_{\text{pos}} = 3.267$). Models emit raw logits; probabilities are obtained by a sigmoid.

**Optimization.** Parameters are optimized with AdamW (learning rate $1\times10^{-3}$, weight decay $1\times10^{-4}$), batch size 32, for up to 80 epochs with a cosine-annealing-with-warm-restarts schedule. The checkpointing and early-stopping criterion is **validation AUROC** (patience 15 epochs, mode = max), and the best-AUROC checkpoint is restored for evaluation. Training used CPU by default for reproducibility across hardware; the random seed was fixed at 42. Key settings are collected in Table 3.

**Table 3. Training configuration (shared across deep models).**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Max epochs | 80 |
| Early-stopping metric | Validation AUROC (patience 15) |
| LR schedule | CosineAnnealingWarmRestarts |
| Loss | Weighted BCE (pos_weight = 3.267) |
| Device / seed | CPU / 42 |

**Table 4. Model-specific hyperparameters.**

| Model | Key hyperparameters | Trainable parameters |
|---|---|---|
| FETA-Transformer | d_model = 64, heads = 4, layers = 2, d_ff = 128, dropout = 0.2 | 87,169 |
| PREG-Net | hidden_dim = 64, GAT layers = 2, heads = 4, dropout = 0.2 | 16,130 |
| Ensemble (average) | fusion = mean of sub-model logits | 103,299 |
| Ensemble (learned) | fusion weights learned (≈ 0.542 / 0.481) | 103,299 |

### 3.9 Explainability

Explainability is treated as a first-class output. For the FETA-Transformer, the temporal-pooling weights $\alpha_t$ quantify how much each scan contributed, and the maternal cross-attention weights quantify where the maternal query attended along the sequence; these are aggregated by gestational week and by outcome. For PREG-Net, per-node importance identifies which clinical variables mattered, and per-edge attention (averaged over layers and heads) identifies which physiological relationships were used; edge attention is compared against the encoded physiological rationale. Both models' attention dictionaries are exported per patient, enabling representative case studies across the four confusion-matrix categories.

### 3.10 Evaluation Protocol and Metrics

Two complementary evaluations are reported. **(i) Single held-out split:** models are trained on the training split, selected on the validation split, and evaluated once on the untouched test split. **(ii) Five-fold cross-validation:** the pipeline is repeated across five stratified folds to estimate the mean and standard deviation of each metric, giving a more robust picture than a single split of only 122 test patients.

Metrics comprise AUROC (the primary metric), AUPRC, accuracy, sensitivity (recall), specificity, precision, and F1 at a 0.5 probability threshold. Percentile bootstrap resampling of the test predictions provides 95 % confidence intervals for AUROC. Metric implementations include pure-NumPy fallbacks and return undefined values gracefully for degenerate single-class resamples, which are skipped rather than allowed to distort estimates.

---

## 4. Results

### 4.1 Experimental Settings

All experiments were implemented in Python using PyTorch for the deep models and scikit-learn for the tabular baselines; XGBoost was included as an optional baseline. Training and evaluation were performed on CPU with a fixed random seed (42) to maximize reproducibility across hardware. Both the single-split and cross-validation protocols used identical preprocessing, the same patient-level stratified splitting, and the same training configuration (Table 3). The deep models were selected on validation AUROC and evaluated once on the held-out test split; the five tabular baselines were fit on engineered per-patient features (see §4.3) and evaluated on the same split.

### 4.2 Dataset Characteristics

The synthetic cohort comprised 800 patients contributing 2,408 scans, with 188 losses (23.5 %) and 612 viable pregnancies (76.5 %). Patients contributed between two and five scans (mean 3.01, median 3), indexed across gestational weeks 5–14. Validation checks confirmed that every patient had at least the minimum number of scans, that gestational ages fell within range, and that no missing ultrasound values remained after imputation. Table 5 summarizes the cohort.

**Table 5. Synthetic cohort summary.**

| Attribute | Value |
|---|---|
| Total patients | 800 |
| Total scans | 2,408 |
| Pregnancy-loss cases | 188 (23.5 %) |
| Viable pregnancies | 612 (76.5 %) |
| Scans per patient | 2–5 (mean 3.01, median 3) |
| Gestational-age range | 5–14 weeks |
| Ultrasound features | FHR, CRL, GS, YSD |
| Maternal features | age, BMI, parity, gravidity, previous loss, conception (IVF/spontaneous), singleton |
| Train / validation / test | 559 / 119 / 122 patients (test: 29 loss, 93 viable) |
| Random seed | 42 |

> **Figure 4 (placeholder).** *Cohort and trajectory visualization.* A 2×2 panel: (a) outcome class balance bar chart; (b) histogram of scans per patient; (c) FHR versus gestational week with viable and loss trajectories overlaid (mean ± band), showing the depressed/declining FHR of loss cases; (d) CRL versus gestational week by outcome. Intended to convey that the synthetic curves are biologically plausible and that loss cases deviate downward.

### 4.3 Single-Split Model Comparison

On the held-out test split, the ensemble models achieved the highest discrimination, followed by the FETA-Transformer and then PREG-Net; all four proposed models were competitive with, and generally ahead of, the five tabular baselines. The tabular baselines were trained on engineered per-patient features (per-channel mean, last value, and slope; scan count; and the seven maternal features), representing the conventional feature-engineering pipeline the deep models are designed to replace. Table 6 reports the full metric set.

**Table 6. Held-out test-set performance (single split), ranked by AUROC. Proposed models in bold.**

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | F1 | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ensemble (average)** | **0.9985** | 0.9954 | 0.9754 | 1.0000 | 0.9677 | 0.9508 | 0.9062 |
| **Ensemble (learned)** | **0.9985** | 0.9958 | 0.9754 | 0.9655 | 0.9785 | 0.9492 | 0.9333 |
| **FETA-Transformer** | **0.9941** | 0.9824 | 0.9508 | 0.8276 | 0.9892 | 0.8889 | 0.9600 |
| Random Forest | 0.9911 | 0.9702 | 0.9426 | 0.9310 | 0.9462 | 0.8852 | 0.8438 |
| **PREG-Net** | **0.9885** | 0.9746 | 0.9426 | 0.9655 | 0.9355 | 0.8889 | 0.8235 |
| XGBoost | 0.9874 | 0.9609 | 0.9426 | 0.9310 | 0.9462 | 0.8852 | 0.8438 |
| Logistic Regression | 0.9870 | 0.9521 | 0.9508 | 0.9310 | 0.9570 | 0.9000 | 0.8710 |
| Multilayer Perceptron | 0.9867 | 0.9593 | 0.9344 | 0.8966 | 0.9462 | 0.8667 | 0.8387 |
| Hist. Gradient Boosting | 0.9863 | 0.9593 | 0.9426 | 0.8621 | 0.9677 | 0.8772 | 0.8929 |

Two patterns are worth noting beyond the ranking. First, the individual deep models exhibit a complementary error profile: the FETA-Transformer is conservative (high specificity 0.989, lower sensitivity 0.828), whereas PREG-Net is more aggressive (higher sensitivity 0.966, lower specificity 0.936). The average ensemble inherits the best of both, reaching sensitivity 1.000 at specificity 0.968 — the signature of a genuinely complementary combination rather than a redundant one, and consistent with the learned fusion weights being close to equal (§3.7). Second, the absolute AUROC values are uniformly very high (all ≥ 0.986), which — as discussed in §5 — reflects the separability of the synthetic cohort more than it reflects a difficult, realistic prediction problem.

> **Figure 5 (placeholder).** *ROC and precision–recall curves on the test split.* Two side-by-side panels. **Left:** ROC curves for all four proposed models (bold) and the five baselines (thin), with the diagonal reference; legend annotated with AUROC. **Right:** precision–recall curves with the class-prevalence baseline; legend annotated with AUPRC. Purpose: show near-ceiling, tightly clustered curves.

### 4.4 Cross-Validation

Because the single test split contains only 122 patients, we repeated the entire pipeline under five-fold stratified cross-validation. Under this more robust protocol the models cluster tightly, and their standard-deviation bands overlap substantially. Notably, a regularized logistic-regression baseline attained the highest mean AUROC, and the proposed deep models — while fully competitive — did not separate from strong baselines in a statistically meaningful way. Table 7 reports mean ± standard deviation across folds.

**Table 7. Five-fold cross-validation performance (mean ± SD), ranked by mean AUROC.**

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression (sklearn) | 0.9930 ± 0.0042 | 0.9806 ± 0.0116 | 0.9600 ± 0.0224 | 0.9415 ± 0.0223 | 0.9657 ± 0.0275 | 0.9185 ± 0.0425 |
| Hist. Gradient Boosting | 0.9889 ± 0.0117 | 0.9713 ± 0.0272 | 0.9600 ± 0.0163 | 0.9043 ± 0.0556 | 0.9771 ± 0.0158 | 0.9138 ± 0.0354 |
| Random Forest | 0.9889 ± 0.0103 | 0.9699 ± 0.0272 | 0.9625 ± 0.0193 | 0.9148 ± 0.0347 | 0.9771 ± 0.0158 | 0.9200 ± 0.0397 |
| XGBoost | 0.9885 ± 0.0119 | 0.9732 ± 0.0255 | 0.9650 ± 0.0180 | 0.9149 ± 0.0344 | 0.9804 ± 0.0170 | 0.9252 ± 0.0373 |
| **Ensemble (learned)** | 0.9873 ± 0.0146 | 0.9733 ± 0.0203 | 0.9675 ± 0.0081 | 0.9307 ± 0.0411 | 0.9788 ± 0.0109 | 0.9306 ± 0.0181 |
| **Ensemble (average)** | 0.9865 ± 0.0133 | 0.9735 ± 0.0156 | 0.9338 ± 0.0387 | 0.9413 ± 0.0518 | 0.9314 ± 0.0628 | 0.8747 ± 0.0612 |
| Multilayer Perceptron | 0.9864 ± 0.0104 | 0.9694 ± 0.0170 | 0.9575 ± 0.0223 | 0.9361 ± 0.0238 | 0.9640 ± 0.0250 | 0.9130 ± 0.0424 |
| **FETA-Transformer** | 0.9859 ± 0.0115 | 0.9685 ± 0.0227 | 0.9587 ± 0.0144 | 0.9201 ± 0.0425 | 0.9706 ± 0.0222 | 0.9134 ± 0.0288 |
| **PREG-Net** | 0.9761 ± 0.0116 | 0.9374 ± 0.0264 | 0.9200 ± 0.0291 | 0.9144 ± 0.0587 | 0.9215 ± 0.0422 | 0.8448 ± 0.0449 |

The contrast between Tables 6 and 7 is itself a result. The single split flatters the ensemble (its top-ranked AUROC of 0.9985 sits above the cross-validated mean of 0.9865–0.9873), while cross-validation reveals that model ordering is not stable and that a linear baseline is at least as good on average. We treat the cross-validated numbers as the more trustworthy estimate of generalization on this cohort.

> **Figure 6 (placeholder).** *Cross-validation forest plot.* A horizontal point-and-error-bar chart of mean AUROC ± SD for all nine models (proposed models highlighted), sorted by mean, with a vertical reference at the best mean. Purpose: make the overlapping error bars and the competitiveness of the logistic baseline visually explicit.

### 4.5 Calibration and Confusion Structure

Confusion matrices for the four proposed models at the 0.5 threshold are consistent with the metric table: the ensemble makes the fewest errors overall, FETA errs toward false negatives (missed losses), and PREG-Net errs toward false positives (false alarms). Bootstrap 95 % confidence intervals for test AUROC are wide and reach the ceiling for most models, reinforcing that decimal-level differences on the single split are not statistically resolvable (Table 8).

**Table 8. Bootstrap 95 % confidence intervals for test-set AUROC (percentile method).**

| Model | AUROC | 95 % CI |
|---|---:|---|
| Ensemble (average) | 0.9985 | [0.9945, 1.0000] |
| Ensemble (learned) | 0.9985 | [0.9939, 1.0000] |
| FETA-Transformer | 0.9941 | [0.9833, 1.0000] |
| Random Forest | 0.9911 | [0.9766, 1.0000] |
| PREG-Net | 0.9885 | [0.9682, 1.0000] |
| XGBoost | 0.9874 | [0.9692, 0.9980] |
| Logistic Regression | 0.9870 | [0.9685, 0.9993] |
| Multilayer Perceptron | 0.9867 | [0.9694, 0.9977] |
| Hist. Gradient Boosting | 0.9863 | [0.9691, 0.9974] |

> **Figure 7 (placeholder).** *Confusion matrices.* A 2×2 grid of 2×2 confusion matrices (FETA-Transformer, PREG-Net, average ensemble, learned ensemble) at threshold 0.5 on the test split, annotated with counts. Purpose: visualize the false-negative-leaning FETA versus false-positive-leaning PREG-Net error profiles and the balanced ensemble.

### 4.6 FETA-Transformer Temporal Attention

Aggregating the FETA-Transformer's temporal-pooling weights by gestational week reveals a coherent, clinically legible pattern: attention mass is low at the earliest scans and rises across the first trimester, concentrating on the mid-to-late window (roughly weeks 10–13) where growth divergence between viable and loss trajectories is most pronounced. For viable pregnancies the mean temporal attention rose from ≈ 0.04–0.05 at weeks 6–7 to ≈ 0.69 at week 11; for loss pregnancies it rose comparably, peaking at the latest observed weeks. The maternal cross-attention showed the complementary tendency of weighting the earliest scans more heavily (mean maternal-to-time attention ≈ 0.38–0.42 at weeks 5–7, declining thereafter), consistent with maternal context being most informative before strong ultrasound evidence accumulates. Representative values are given in Table 9.

**Table 9. FETA-Transformer mean temporal-pooling attention by gestational week and outcome (test split; selected weeks).**

| Outcome | Week | Scans | Mean temporal attention | Mean maternal→time attention |
|---|---:|---:|---:|---:|
| Viable | 7 | 49 | 0.051 | 0.389 |
| Viable | 9 | 47 | 0.412 | 0.323 |
| Viable | 11 | 30 | 0.694 | 0.278 |
| Viable | 12 | 12 | 0.641 | 0.264 |
| Loss | 8 | 12 | 0.345 | 0.364 |
| Loss | 10 | 12 | 0.457 | 0.320 |
| Loss | 12 | 6 | 0.558 | 0.290 |
| Loss | 14 | 3 | 0.699 | 0.296 |

> **Figure 8 (placeholder).** *FETA temporal attention by gestational week.* Two line panels split by outcome (viable vs loss). **Left:** mean temporal-pooling attention versus gestational week. **Right:** mean maternal→time cross-attention versus gestational week. Purpose: show attention rising toward later scans while maternal attention concentrates early.

### 4.7 PREG-Net Node and Edge Importance

Averaged over the test split, PREG-Net assigned by far the greatest node importance to **fetal heart rate** (mean 0.228), followed by **yolk-sac diameter** (0.061), then the maternal nodes *singleton* (0.040) and *gravidity* (0.030) and the ultrasound node GS (0.019); CRL, age, BMI, prior loss, IVF, and parity received small weights. The prominence of FHR and YSD is clinically reasonable, as both are recognized early markers of embryonic viability. Table 10 lists the ranking.

**Table 10. PREG-Net mean node importance (test split).**

| Rank | Node | Family | Mean importance |
|---:|---|---|---:|
| 1 | FHR | ultrasound | 0.2283 |
| 2 | YSD | ultrasound | 0.0613 |
| 3 | Singleton | maternal | 0.0400 |
| 4 | Gravidity | maternal | 0.0301 |
| 5 | GS | ultrasound | 0.0189 |
| 6 | Conception (IVF) | maternal | 0.0036 |
| 7 | Previous loss | maternal | 0.0032 |
| 8 | BMI | maternal | 0.0030 |
| 9 | CRL | ultrasound | 0.0029 |
| 10 | Age | maternal | 0.0022 |
| 11 | Parity | maternal | 0.0014 |

Edge attention was distributed across all three edge families. The highest-attention edges were physiological growth relationships (CRL→GS 0.217, FHR→YSD 0.195, GS→YSD 0.168) and same-variable temporal links (CRL and GS across consecutive scans, ≈ 0.15–0.17). Among the edges we had labeled as the *primary* research rationale, previous-loss→YSD (0.163) and FHR→CRL (0.138) received moderate attention. Importantly — and contrary to a simple expectation that the model would weight the three named "primary rationale" edges most heavily — several *additional* physiological and temporal edges attracted higher attention than the primary ones (Table 11). We report this partial, rather than clean, concordance honestly: the model does concentrate on physiologically meaningful growth and temporal relationships, but not exclusively on the specific directed edges emphasized in the design rationale.

**Table 11. PREG-Net edge attention (test split; top edges and selected primary-rationale edges).**

| Relationship | Edge family | Mean attention |
|---|---|---:|
| CRL → GS | Physiological (additional) | 0.2167 |
| FHR → YSD | Physiological (additional) | 0.1952 |
| CRL → CRL (temporal, backward) | Temporal | 0.1680 |
| GS → YSD | Physiological (additional) | 0.1679 |
| Previous loss → YSD | **Primary rationale** | 0.1626 |
| FHR → CRL | **Primary rationale** | 0.1382 |

> **Figure 9 (placeholder).** *PREG-Net node importance and edge attention.* Two panels. **Left:** horizontal bar chart of mean node importance (Table 10), colored by family (ultrasound vs maternal). **Right:** either a bar chart of mean edge attention grouped by edge family, or the knowledge graph with edges shaded by mean attention. Purpose: highlight FHR/YSD dominance and the distribution of attention across physiological, temporal, and maternal edges.

### 4.8 Case Studies

Per-patient explanations illustrate how the two models reason and, importantly, disagree. In a representative **true positive** (a loss correctly flagged), both models were near-certain (PREG-Net probability 1.000, FETA 1.000). In a **true negative**, both were near-zero (0.000 / 0.001). The **false positive** was a viable pregnancy that both models over-called, with PREG-Net especially confident (0.998 vs FETA 0.850). The most instructive case is a **false negative** in which the two views split sharply: PREG-Net assigned a low probability (0.137) and missed the loss, whereas FETA correctly flagged it (0.977) — a concrete example of complementary failure modes and of why late fusion helps. Table 12 summarizes these cases.

**Table 12. Representative per-patient case studies (test split).**

| Case type | Patient | True label | PREG-Net prob. | FETA prob. |
|---|---|---:|---:|---:|
| True positive | P00548 | Loss | 1.000 | 1.000 |
| True negative | P00320 | Viable | 0.000 | 0.001 |
| False positive | P00089 | Viable | 0.998 | 0.850 |
| False negative | P00099 | Loss | 0.137 | 0.977 |

---

## 5. Discussion

**Principal findings.** This study set out to implement and evaluate two complementary, interpretable architectures for first-trimester loss prediction and to test whether combining a temporal view and a relational view is beneficial. Three findings stand out. First, both models are implementable in pure PyTorch, are lightweight (16k–103k parameters), and behave as designed: FETA learns a sensible gestational-week attention profile, and PREG-Net produces node/edge attributions dominated by physiologically plausible signals (FHR, YSD, growth relationships). Second, the two models have genuinely complementary error profiles — conservative FETA versus aggressive PREG-Net — and their late fusion produces the best single-split operating point, with the learned fusion weighting the two views almost equally. Third, and most importantly for honest reporting, this apparent superiority does not survive cross-validation: under five folds all models cluster within overlapping error bands and a regularized logistic-regression baseline is at least as accurate on average.

**Interpreting the very high scores.** The uniformly high AUROC values (≥ 0.986 on the single split; 0.976–0.993 under cross-validation) should be read as a property of the *data*, not of the models. The synthetic cohort is generated from smooth, literature-derived growth curves with outcome-dependent latent factors, which makes viable and loss trajectories more cleanly separable than real clinical measurements — with their noise, comorbidities, and label ambiguity — ever are. That even a linear model reaches ≈ 0.99 AUROC is direct evidence that the task, as instantiated here, is easy. The value of the experiment therefore lies in the *methodology and interpretability*, not in the headline numbers.

**Why explainability matters here.** The attention analyses convert both models from black boxes into inspectable ones. FETA's rising temporal attention toward mid-to-late first-trimester scans, and its early emphasis on maternal context, are the kinds of statements a clinician can evaluate against domain knowledge. PREG-Net's node ranking (FHR and YSD at the top) and its edge attention over growth and temporal relationships likewise expose the model's reasoning. We deliberately did not overclaim here: the primary-rationale edges did not dominate the edge attention, and we reported that discrepancy rather than smoothing it over. This kind of transparency is precisely what the introduction identified as missing from much of the prior literature.

**Relation to prior work.** Compared with the studies in Table 1, the present framework's methodological novelty is the pairing of learned temporal attention over irregular sequences with an explicitly knowledge-guided relational model, evaluated head-to-head under one protocol. Prior work has demonstrated strong results with automated biometric extraction, radiomics, and multimodal fusion, often with real cohorts and — critically — with external validation (for example the multi-site elastic-net model of Murugesu et al.). Our study does not match that external-validity bar; it contributes instead a reproducible, fully documented, interpretable modeling pipeline that could be transferred to real data.

**Practical implications.** For a deployment on data like this, the pragmatic conclusion is that a simple, well-regularized baseline would be preferred on grounds of parsimony and stability, and the deep models would need to prove their worth on harder, noisier, real-world data where learned temporal and relational structure is more likely to pay off. The complementary-error result does, however, suggest that if the individual models were to become competitive on real data, ensembling them would be a productive strategy.

---

## 6. Limitations

This work has several important limitations, which we state plainly.

1. **The data are entirely synthetic.** No real patients are involved. The cohort is generated from literature-informed growth curves and latent-factor models; it does not draw from, and has not been validated against, any real dataset. All reported performance therefore measures how well the models recover structure that the generator built in — it is *not* clinical evidence of predictive value, and the numbers must not be interpreted as such.

2. **Strong baselines match the proposed models.** Under cross-validation, a regularized logistic regression achieved the highest mean AUROC, and the deep models did not separate from tabular baselines within their error bars. On this cohort, architectural sophistication is not justified by performance.

3. **Ceiling effects and small test size.** The task's high separability produces near-ceiling metrics and wide bootstrap confidence intervals (many reaching 1.0), and the single test split contains only 122 patients. Decimal-level comparisons on the single split are not statistically resolvable.

4. **No external validation, calibration analysis, or decision-curve analysis.** Unlike several comparators, we do not report performance on an independent cohort, formal calibration (for example reliability curves, calibration slope/intercept, Brier decomposition), or clinical net benefit. These are prerequisites for any clinical claim.

5. **Partial concordance of learned attention with the encoded rationale.** PREG-Net's highest-attention edges were not the specific directed edges designated as the primary physiological rationale, indicating that interpretability claims should be made carefully and that the knowledge graph's design could be refined.

6. **Ablations not performed.** We did not systematically ablate the individual architectural components (continuous positional encoding, modality-specific projections, attention pooling, cross-attention; graph edge families and attention heads). Consequently we cannot attribute performance to specific design choices, and any component-level claims would be speculative.

7. **Fixed knowledge graph and limited temporal depth.** The graph structure is hand-specified and static across patients (only its attention weights vary), and sequences are truncated to a maximum of five scans; both choices constrain the models.

---

## 7. Conclusion

We presented and evaluated two complementary, interpretable deep-learning architectures for first-trimester pregnancy-loss prediction: the FETA-Transformer, which learns temporal representations of irregular longitudinal ultrasound through continuous positional encoding, modality-specific projections, attention pooling, and maternal cross-attention; and PREG-Net, a knowledge-guided graph-attention network that models physiological relationships explicitly and yields node- and edge-level attributions. A late-fusion ensemble combined the temporal and relational views. On a fully synthetic cohort of 800 patients, the ensemble achieved the best single-split discrimination (test AUROC 0.9985), the individual models displayed clearly complementary error profiles, and attention analysis produced clinically legible explanations — FETA concentrating on mid-to-late first-trimester scans and PREG-Net elevating fetal heart rate and yolk-sac diameter. At the same time, cross-validation showed all models clustered within overlapping error bands, with a simple logistic-regression baseline fully competitive, and the entire study rests on simulated rather than real data. We therefore frame this work honestly as a **methodological and reproducibility contribution**: it delivers a documented, interpretable, end-to-end pipeline and a candid appraisal of where sophisticated architectures do and do not help. The essential next steps are validation on real, consented, multi-center longitudinal cohorts; formal calibration and decision-curve analysis; and component ablations to establish which design choices carry their weight. Only after those steps could any claim of clinical utility be responsibly advanced.

---

## Data and Code Availability

All data used in this study are synthetic and are regenerable deterministically from the released generator under a fixed seed (seed = 42); no real patient data were used or distributed. The synthetic-cohort generator, the model implementations (FETA-Transformer, PREG-Net, and the ensemble, in pure PyTorch), the training and evaluation scripts, the cross-validation and explainability pipelines, and a single self-contained Jupyter notebook that reproduces the full workflow end-to-end are available in the project repository. Derived result artifacts (metric tables, cross-validation summaries, and explainability exports) are also included.

## Ethics Statement

This study did not involve human participants, human tissue, or identifiable personal data. All analyses were performed on computationally generated synthetic data; accordingly, institutional review board approval and informed consent were not applicable. Because the framework is intended for eventual clinical translation, we emphasize that no results reported here constitute clinical evidence, and that prospective ethical review, consent procedures, and regulatory assessment would be required before any use on real patient data.

## Conflict of Interest

The authors declare no conflict of interest.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## References

Note: The framework builds directly on the conceptual model designs described in the project's methodology document (FETA-Transformer and PREG-Net) and is positioned against the comparator literature below. References [15]–[23] correspond to the studies discussed in Section 2 and Table 1.

[1] Mikołaj KW, Christensen AN, Taksøe-Vester CA, et al. Predicting abnormal fetal growth using deep learning. *NPJ Digital Medicine*. 2025;8(1):318.

[2] Logeshwaran J, Yuvaraj N, Sharma V, Shukla RP, Kumar D. A meta-learning approach for improving medical image segmentation with transfer learning. *2024 International Conference on Recent Innovation in Smart and Sustainable Technology (ICRISST)*. IEEE; 2024:1–6.

[3] Bhavani G, Jeyalakshmi C. Prediction of clinical risk factors in pregnancy using optimized neural network scheme. *Placenta*. 2025;163:33–42.

[4] Kakani TA, Vedula J, Mohammed M, Gupta R, Hudani K, Yuvaraj N. Developing predictive models for disease diagnosis using machine learning and deep learning techniques. *2025 6th International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV)*. IEEE; 2025:158–163.

[5] Pavanya M, Chadaga K, Vasudeva A, Rao BK, Prabhu S, Bhat SK. Prediction of birthweight with early and mid-pregnancy antenatal markers utilising machine learning and explainable artificial intelligence. *Scientific Reports*. 2025;15(1):26223.

[6] Gupta R, Kakani TA, Vedula J, Mohammed M, Hudani K, Yuvaraj N. Advancing clinical decision-making using artificial intelligence and machine learning for accurate disease diagnosis. *2025 6th International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV)*. IEEE; 2025:164–169.

[7] Ferreira I, Simões J, Pereira B, Correia J, Areia AL. Ensemble learning for fetal ultrasound and maternal–fetal data to predict mode of delivery after labor induction. *Scientific Reports*. 2024;14(1):15275.

[8] Mennickent D, Rodríguez A, Opazo MC, et al. Machine learning applied in maternal and fetal health: a narrative review focused on pregnancy diseases and complications. *Frontiers in Endocrinology*. 2023;14:1130139.

[9] Hua Q, Yang F, Zhou Y, Shi F, You X, Guo J, Li L. Predictive models using machine learning to identify fetal growth restriction in patients with preeclampsia: development and evaluation study. *Journal of Medical Internet Research*. 2025;27:e70068.

[10] Bertini A, Salas R, Chabert S, Sobrevia L, Pardo F. Using machine learning to predict complications in pregnancy: a systematic review. *Frontiers in Bioengineering and Biotechnology*. 2022;9:780389.

[11] Wu S, Dong J, Shi J, et al. Machine learning prediction of short cervix in mid-pregnancy based on multimodal data from the first-trimester screening period: an observational study in a high-risk population. *Biomedicines*. 2025;13(9):2057.

[12] Wang C, Johansson ALV, Nyberg C, et al. Prediction of pregnancy-related complications in women undergoing assisted reproduction, using machine learning methods. *Fertility and Sterility*. 2024;122(1):95–105.

[13] Zhu Z, Wei N, Guo J, et al. Predicting the risk of threatened abortion using machine learning methods: a comparative study. *BMC Pregnancy and Childbirth*. 2025;25(1):901.

[14] Zhang M, Sheng J. The application of super-resolution ultrasound radiomics models in predicting the failure of conservative treatment for ectopic pregnancy. *Reproductive Biology and Endocrinology*. 2025;23(1):102.

[15] Liu L, Zang Y, Zheng H, et al. An AI method to predict pregnancy loss by extracting biological indicators from embryo ultrasound recordings in early pregnancy. *Scientific Reports*. 2025;15(1):25946.

[16] Liu C, Wei X, Wang F. The predictive value of ultrasound markers for pregnancy outcomes in recurrent pregnancy loss: a retrospective study. *Scientific Reports*. 2024;14(1):16657.

[17] Yan S, Xiong F, Xin Y, Zhou Z, Liu W. Optimizing evaluation of endometrial receptivity in recurrent pregnancy loss: a preliminary investigation integrating radiomics from multimodal ultrasound via machine learning. *Frontiers in Endocrinology*. 2024;15:1380829.

[18] Yan S, Xiong F, Xin Y, Zhou Z, Liu W. Automated assessment of endometrial receptivity for screening recurrent pregnancy loss risk using deep learning-enhanced ultrasound and clinical data. *Frontiers in Physiology*. 2024;15:1404418.

[19] Zhang J, Liu S, Rong Y, et al. Prediction of uterine cavity conception environment using two-dimensional transvaginal ultrasound imaging semantic feature-based machine learning: a case-control study. *BMC Pregnancy and Childbirth*. 2025;25(1):1204.

[20] Murugesu S, Linton-Reid K, Barcroft J, et al. Reducing uncertainty in early pregnancy: using clinical features and radiomics to develop a machine learning model to predict outcome. *Human Reproduction*. 2025;40(Suppl 1):deaf097-219.

[21] Yang X, Wang R, Zhang W, Yang Y, Wang F. Predicting risk of the subsequent early pregnancy loss in women with recurrent pregnancy loss based on preconception data. *BMC Women's Health*. 2024;24(1):381.

[22] Lee LH, Bradburn E, Craik R, et al. Machine learning for accurate estimation of fetal gestational age based on ultrasound images. *NPJ Digital Medicine*. 2023;6(1):36.

[23] Murugesu S, Linton-Reid K, Braun E, et al. Predicting outcomes of expectant and medical management in early pregnancy miscarriage using machine learning to develop and validate multivariable clinical prediction models. *BMC Pregnancy and Childbirth*. 2025;25(1):225.

[24] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in Neural Information Processing Systems*. 2017;30:5998–6008.

[25] Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. *International Conference on Learning Representations (ICLR)*. 2018.

---

*Manuscript prepared as a methodological and reproducibility study. All quantitative results were generated from the accompanying synthetic-data pipeline; figures are described as placeholders for insertion of the corresponding generated plots.*
