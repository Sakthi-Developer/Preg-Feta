# Methods Draft

## Study Design

We conducted a model-development and evaluation study using a synthetic first-trimester longitudinal ultrasound cohort. The objective was to compare an interpretable graph attention model, PREG-Net, with a temporal Transformer model, FETA-Transformer, and to evaluate whether late fusion of graph and temporal representations improved pregnancy loss prediction. The study was designed as a proof of concept for a future clinical cohort rather than as a clinically validated diagnostic model.

## Synthetic Cohort Construction

The cohort was generated with `synthetic_generator_v2.py` using seed 42. The final dataset contained 800 patients, 2408 scan records, 612 ongoing pregnancies, and 188 pregnancy losses, corresponding to a 23.5 percent loss rate. Each patient had 2-5 first-trimester ultrasound scans. Patient-level variables included maternal age, BMI, parity, gravidity, previous pregnancy loss, conception method, singleton status, and outcome label.

Longitudinal ultrasound variables were fetal heart rate (FHR), crown-rump length (CRL), gestational sac diameter (GS), and yolk sac diameter (YSD). Values were generated from biologically motivated first-trimester curves with patient-level latent factors for growth, cardiac function, placentation, and measurement noise. Pregnancy-loss cases were modeled with overlapping but shifted latent-factor distributions, allowing decelerating or abnormal trajectories without making classification deterministic. Consecutive scans were generated autoregressively so that each measurement depended partly on the previous scan and partly on the expected value at the current gestational age.

Clinically plausible missingness was introduced as a function of gestational age and outcome. FHR had higher missingness before 6.5 weeks, CRL was sometimes missing very early, GS was rarely missing, and YSD was more likely to be missing very early or after 11 weeks. Missing ultrasound values were interpolated within patient, then back-filled and forward-filled, with residual missing values filled by feature medians. The saved generated dataset had no remaining missing ultrasound values.

## Etiologic Heterogeneity: The Dual-Channel Loss Model

The experiments in this work use a *dual-etiology* configuration of the generator (`--etiology dual`), and we disclose this design explicitly because it is central to interpreting the fusion results. First-trimester pregnancy loss is etiologically heterogeneous: some losses are embryonic and manifest in the longitudinal scan trajectory (decelerating crown-rump-length or gestational-sac growth, bradycardia, an enlarged yolk sac), whereas others are maternal or constitutional and present with a near-normal early scan while being driven by an interaction of maternal risk factors (advanced age compounded by elevated BMI and prior loss). We modelled this heterogeneity directly. Each simulated pregnancy is assigned two independent latent loss channels — a **sonographic channel**, whose severity drives the scan trajectory that FETA-Transformer observes, and a **maternal channel**, whose risk is an interaction of the maternal variables that PREG-Net observes — and the observed outcome is their logical OR (a loss occurs if either mechanism fires). A shared frailty term (`--channel-corr`) and a directed maternal-to-sonographic coupling (`--sono-mat-coupling`) produce a controlled fraction of "both-channel" losses that are visible to both modalities; the remainder are single-channel losses visible to only one. Under the shipped balanced setting, roughly three quarters of losses are single-channel, so genuine complementary information exists that no single-modality model can fully recover.

This design is a deliberate modelling assumption, not a measured property of a real cohort. Its purpose is to construct an in-silico testbed in which multimodal fusion is *informative* — the legacy single-outcome generator makes both modalities noisy views of the same label, so the two single-modality models make highly correlated errors and fusion has little to add, which understates rather than tests the fusion hypothesis. Because each modality here reflects only one channel, the fusion gains we report are a demonstration that the architecture can exploit etiologic heterogeneity when it is present; they are not evidence about the magnitude of any effect in real pregnancies. This limitation is restated in the Limitations section.

## Complementary-Specialist Configuration

To match the dual-channel design, the two towers are configured as complementary specialists rather than as two models of the same inputs. FETA-Transformer is the ultrasound specialist and reads only the longitudinal scan sequence. PREG-Net is the clinical specialist and is built over a knowledge graph of the seven maternal variables only (age, BMI, parity, gravidity, previous loss, conception method, singleton status), with no ultrasound nodes; intra-maternal edges connect clinically related risk factors (for example age-BMI, age-prior-loss, gravidity-parity) so the graph attention can capture the maternal interaction structure that drives the maternal channel. Each specialist therefore observes exactly one of the two loss channels, which is what allows their fusion to recover losses that neither sees alone. This is selected at training time with the `--clinical-only` flag on `train_preg.py`, `train_hybrid.py`, and `cross_validate.py`.

## Preprocessing And Split Strategy

The dataset was represented as a fixed-length sequence of up to five scans per patient. For FETA-Transformer, each patient was converted into temporal features with shape `(T, 4)`, gestational ages with shape `(T,)`, a temporal mask indicating real versus padded scans, maternal features with shape `(7,)`, and a binary label. For PREG-Net, the same batch was converted into a graph with temporal ultrasound nodes and static maternal nodes.

Single-split experiments used a stratified 70/15/15 train/validation/test split with seed 42. Training-only normalization statistics were computed for continuous ultrasound and maternal variables and applied to validation and test data. Binary maternal features were kept in their original 0/1 form. Five-fold cross-validation used stratified folds, with a validation subset drawn from each training fold. Fold identifiers were saved under `results/cross_validation/splits/`.

## PREG-Net

PREG-Net was the primary model. It represented each patient as a graph with temporal ultrasound nodes for FHR, CRL, GS, and YSD at each scan time point, plus static maternal nodes for age, BMI, parity, gravidity, previous loss, conception method, and singleton status. Edges encoded clinical relationships and temporal continuity. Intra-time ultrasound edges included FHR-to-CRL, YSD-to-GS, CRL-to-GS, and FHR-to-YSD with bidirectional counterparts. Maternal-to-ultrasound edges connected age, BMI, previous loss, gravidity, parity, and conception method to selected ultrasound nodes. Temporal edges connected the same ultrasound variable across consecutive scans in both directions.

The model encoded each node with a scalar value projection plus node-type embedding. Stacked graph attention layers performed patient-specific message passing over the knowledge-guided graph. A graph attention readout produced a patient-level representation and node importance scores. A classification head produced the pregnancy-loss logit. PREG-Net explainability outputs included node importance, edge attention averaged across layers and heads, clinical edge-group summaries, and patient-specific graph visualizations.

## FETA-Transformer

FETA-Transformer was the temporal comparison model. It accepted the longitudinal ultrasound sequence, gestational ages, temporal mask, and maternal features. Each ultrasound modality was projected separately into a shared model dimension. Continuous positional encoding used actual gestational age rather than scan index, allowing the model to represent irregular scan timing. Transformer encoder layers modeled temporal interactions across scans. Attention-based temporal pooling produced scan-level attention weights that were exported for interpretation. Maternal cross-attention conditioned the temporal representation on static maternal features before classification.

The current trained architecture applies maternal cross-attention to the pooled temporal representation. Therefore, exported maternal-to-time maps are post-hoc interpretability maps derived from the trained maternal query and temporal keys, not a replacement for a fully retrained maternal-to-token cross-attention architecture.

## Hybrid Cross-Attention Fusion

The primary fusion model is the Hybrid cross-attention predictor, an intermediate-fusion architecture in which the two specialists exchange information mid-network before either commits to a prediction. FETA-Transformer encodes its per-scan token sequence and PREG-Net encodes its per-node embeddings; a bidirectional masked cross-attention block then lets the temporal tokens attend to the maternal nodes and vice versa. In the gated configuration used here, each branch's auxiliary head reads its own pure (pre-fusion) representation, so the two towers remain individually predictive and do not collapse into copies of one another, while the cross-attended representations drive a per-patient gate that routes between the two branch predictions and adds a small correction term. This gating is the mechanism by which the model can defer to the ultrasound specialist for sonographic-channel losses and to the clinical specialist for maternal-channel losses on a patient-by-patient basis. An auxiliary per-branch loss (weight 0.3) is added to the fused binary cross-entropy to keep both branches calibrated during joint training. The towers may optionally be warm-started from the separately trained specialist checkpoints; we report both warm-started and end-to-end-trained variants. The model exposes the temporal, node, edge, and cross-modal attention maps and the per-patient gate value for interpretation.

## Ensemble Models

Late-fusion ensembles are reported as fusion baselines against the intermediate-fusion Hybrid. They combined FETA-Transformer and PREG-Net predictions after each tower committed to a logit. The simple ensemble averaged logits from the two component models. The learned-fusion ensemble used a linear layer over the two component logits, initialized to equal weighting and trained end to end.

## Baseline Models

Tabular baselines used engineered patient-level features derived from the same saved splits. Features included scan count, first and latest gestational age, gestational-age span, maternal features, and for each ultrasound variable the latest value, mean value, and slope over gestational age. Baselines included Torch logistic regression, Torch MLP, sklearn logistic regression, Random Forest, HistGradientBoosting, and XGBoost. All baselines were evaluated on the same fixed test split and included in the cross-validation analysis where available.

## Training

Deep models used weighted binary cross-entropy, AdamW optimization, cosine warm restarts, validation AUROC monitoring, early stopping, and checkpointing. The standard configuration used batch size 32, learning rate 0.001, weight decay 0.0001, and patience 15. PREG-Net used hidden dimension 64, two graph attention layers, four heads, and dropout 0.2. FETA-Transformer used model dimension 64, four heads, two Transformer layers, feed-forward dimension 128, and dropout 0.2.

## Evaluation And Statistical Analysis

Performance metrics were AUROC, AUPRC, accuracy, sensitivity, specificity, F1, precision, and recall. Fixed-split model comparison files were saved under `results/model_comparison.*`. Five-fold cross-validation summaries were saved under `results/cross_validation`. Bootstrap 95 percent confidence intervals were computed from fixed test predictions with 2000 bootstrap iterations. Exact McNemar tests compared paired classification correctness. Paired bootstrap AUROC-difference tests were used as a documented alternative to DeLong testing. Calibration was summarized using Brier score, expected calibration error, maximum calibration error, and calibration-curve bins.

## Figure And Table Generation

Paper artifacts were generated under `results/paper_artifacts`. Figures included architecture diagrams, ROC curves, precision-recall curves, confusion matrices, calibration curves, attention heatmaps, PREG-Net node and edge importance summaries, patient graph visualizations, and trajectory overlays. Tables included dataset demographics, fixed-split model metrics, cross-validation summaries, bootstrap confidence intervals, hyperparameters, computational cost, and ablation status.

