# Methods Draft

## Study Design

We conducted a model-development and evaluation study using a synthetic first-trimester longitudinal ultrasound cohort. The objective was to compare an interpretable graph attention model, PREG-Net, with a temporal Transformer model, FETA-Transformer, and to evaluate whether late fusion of graph and temporal representations improved pregnancy loss prediction. The study was designed as a proof of concept for a future clinical cohort rather than as a clinically validated diagnostic model.

## Synthetic Cohort Construction

The cohort was generated with `synthetic_generator_v2.py` using seed 42. The final dataset contained 800 patients, 2408 scan records, 612 ongoing pregnancies, and 188 pregnancy losses, corresponding to a 23.5 percent loss rate. Each patient had 2-5 first-trimester ultrasound scans. Patient-level variables included maternal age, BMI, parity, gravidity, previous pregnancy loss, conception method, singleton status, and outcome label.

Longitudinal ultrasound variables were fetal heart rate (FHR), crown-rump length (CRL), gestational sac diameter (GS), and yolk sac diameter (YSD). Values were generated from biologically motivated first-trimester curves with patient-level latent factors for growth, cardiac function, placentation, and measurement noise. Pregnancy-loss cases were modeled with overlapping but shifted latent-factor distributions, allowing decelerating or abnormal trajectories without making classification deterministic. Consecutive scans were generated autoregressively so that each measurement depended partly on the previous scan and partly on the expected value at the current gestational age.

Clinically plausible missingness was introduced as a function of gestational age and outcome. FHR had higher missingness before 6.5 weeks, CRL was sometimes missing very early, GS was rarely missing, and YSD was more likely to be missing very early or after 11 weeks. Missing ultrasound values were interpolated within patient, then back-filled and forward-filled, with residual missing values filled by feature medians. The saved generated dataset had no remaining missing ultrasound values.

## Preprocessing And Split Strategy

The dataset was represented as a fixed-length sequence of up to five scans per patient. For FETA-Transformer, each patient was converted into temporal features with shape `(T, 4)`, gestational ages with shape `(T,)`, a temporal mask indicating real versus padded scans, maternal features with shape `(7,)`, and a binary label. For PREG-Net, the same batch was converted into a graph with temporal ultrasound nodes and static maternal nodes.

Single-split experiments used a stratified 70/15/15 train/validation/test split with seed 42. Training-only normalization statistics were computed for continuous ultrasound and maternal variables and applied to validation and test data. Binary maternal features were kept in their original 0/1 form. Five-fold cross-validation used stratified folds, with a validation subset drawn from each training fold. Fold identifiers were saved under `results/cross_validation/splits/`.

## PREG-Net

PREG-Net was the primary model. It represented each patient as a graph with temporal ultrasound nodes for FHR, CRL, GS, and YSD at each scan time point, plus static maternal nodes for age, BMI, parity, gravidity, previous loss, conception method, and singleton status. Edges encoded clinical relationships and temporal continuity. Intra-time ultrasound edges included FHR-to-CRL, YSD-to-GS, CRL-to-GS, and FHR-to-YSD with bidirectional counterparts. Maternal-to-ultrasound edges connected age, BMI, previous loss, gravidity, parity, and conception method to selected ultrasound nodes. Temporal edges connected the same ultrasound variable across consecutive scans in both directions.

The model encoded each node with a scalar value projection plus node-type embedding. Stacked graph attention layers performed patient-specific message passing over the knowledge-guided graph. A graph attention readout produced a patient-level representation and node importance scores. A classification head produced the pregnancy-loss logit. PREG-Net explainability outputs included node importance, edge attention averaged across layers and heads, clinical edge-group summaries, and patient-specific graph visualizations.

## FETA-Transformer

FETA-Transformer was the temporal comparison model. It accepted the longitudinal ultrasound sequence, gestational ages, temporal mask, and maternal features. Each ultrasound modality was projected separately into a shared model dimension. Continuous positional encoding used actual gestational age rather than scan index, allowing the model to represent irregular scan timing. Transformer encoder layers modeled temporal interactions across scans. Attention-based temporal pooling produced scan-level attention weights that were exported for interpretation. Maternal cross-attention conditioned the temporal representation on static maternal features before classification.

The current trained architecture applies maternal cross-attention to the pooled temporal representation. Therefore, exported maternal-to-time maps are post-hoc interpretability maps derived from the trained maternal query and temporal keys, not a replacement for a fully retrained maternal-to-token cross-attention architecture.

## Ensemble Models

Late-fusion ensembles combined FETA-Transformer and PREG-Net predictions. The simple ensemble averaged logits from the two component models. The learned-fusion ensemble used a linear layer over the two component logits, initialized to equal weighting and trained end to end. The learned-fusion run reported effective fusion weights of approximately 0.542 for FETA and 0.481 for PREG-Net.

## Baseline Models

Tabular baselines used engineered patient-level features derived from the same saved splits. Features included scan count, first and latest gestational age, gestational-age span, maternal features, and for each ultrasound variable the latest value, mean value, and slope over gestational age. Baselines included Torch logistic regression, Torch MLP, sklearn logistic regression, Random Forest, HistGradientBoosting, and XGBoost. All baselines were evaluated on the same fixed test split and included in the cross-validation analysis where available.

## Training

Deep models used weighted binary cross-entropy, AdamW optimization, cosine warm restarts, validation AUROC monitoring, early stopping, and checkpointing. The standard configuration used batch size 32, learning rate 0.001, weight decay 0.0001, and patience 15. PREG-Net used hidden dimension 64, two graph attention layers, four heads, and dropout 0.2. FETA-Transformer used model dimension 64, four heads, two Transformer layers, feed-forward dimension 128, and dropout 0.2.

## Evaluation And Statistical Analysis

Performance metrics were AUROC, AUPRC, accuracy, sensitivity, specificity, F1, precision, and recall. Fixed-split model comparison files were saved under `results/model_comparison.*`. Five-fold cross-validation summaries were saved under `results/cross_validation`. Bootstrap 95 percent confidence intervals were computed from fixed test predictions with 2000 bootstrap iterations. Exact McNemar tests compared paired classification correctness. Paired bootstrap AUROC-difference tests were used as a documented alternative to DeLong testing. Calibration was summarized using Brier score, expected calibration error, maximum calibration error, and calibration-curve bins.

## Figure And Table Generation

Paper artifacts were generated under `results/paper_artifacts`. Figures included architecture diagrams, ROC curves, precision-recall curves, confusion matrices, calibration curves, attention heatmaps, PREG-Net node and edge importance summaries, patient graph visualizations, and trajectory overlays. Tables included dataset demographics, fixed-split model metrics, cross-validation summaries, bootstrap confidence intervals, hyperparameters, computational cost, and ablation status.

