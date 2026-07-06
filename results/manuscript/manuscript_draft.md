# Knowledge-Guided Graph Attention and Temporal Transformer Models for First-Trimester Pregnancy Loss Prediction

## Abstract

### Background

Early pregnancy loss prediction requires interpretation of longitudinal fetal ultrasound measurements in the context of maternal risk factors. The task is time-sensitive, multimodal, and clinically interdependent: fetal heart rate, crown-rump length, gestational sac diameter, yolk sac diameter, gestational age, and maternal factors all contribute to risk assessment.

### Objective

This study developed and evaluated PREG-Net, a knowledge-guided graph attention model, as the primary interpretable architecture for first-trimester pregnancy loss prediction. FETA-Transformer was evaluated as a temporal comparison model for learning gestational-age-dependent ultrasound trajectories. Late-fusion ensembles assessed whether graph and temporal representations provided complementary predictive signal.

### Methods

Experiments used a synthetic first-trimester cohort of 800 patients and 2408 scans. The cohort included 188 pregnancy losses and 612 ongoing pregnancies. Each patient had 2-5 first-trimester ultrasound scans containing FHR, CRL, GS, and YSD, plus maternal features including age, BMI, parity, gravidity, previous loss, conception method, and singleton status. Models were evaluated on a fixed stratified train/validation/test split and with 5-fold stratified cross-validation. Statistical analysis included bootstrap 95 percent confidence intervals, exact McNemar tests, paired bootstrap AUROC-difference comparisons, and calibration metrics.

### Results

On the fixed test split, PREG-Net achieved AUROC 0.9855, AUPRC 0.9568, accuracy 0.9508, sensitivity 0.9655, specificity 0.9462, and F1 0.9032. FETA-Transformer achieved AUROC 0.9896 and AUPRC 0.9728. Simple and learned ensembles achieved AUROC 0.9933 and AUPRC approximately 0.981. Tabular baselines were highly competitive, with the MLP achieving the highest fixed-split AUROC of 0.9985. In 5-fold cross-validation, sklearn logistic regression had the highest mean AUROC of 0.9930 +/- 0.0042, while learned ensemble had the strongest mean F1 among the research deep models at 0.9306 +/- 0.0181. PREG-Net explanations identified FHR as the highest-importance node family and highlighted clinically meaningful graph relationships as well as additional implementation edges.

### Conclusions

PREG-Net provides an interpretable graph-based framework for modeling clinical relationships in first-trimester pregnancy loss prediction, while FETA-Transformer provides a complementary temporal-attention comparison model. Results support the feasibility of both approaches in a synthetic cohort, but the strong performance of tabular baselines and the absence of external validation require cautious interpretation. Clinical validation, ablation studies, and refinement of maternal-to-temporal attention are necessary before claims of clinical utility.

## Introduction

Early pregnancy loss assessment depends on integrating serial ultrasound measurements with maternal risk factors. FHR, CRL, GS, and YSD are interpreted relative to gestational age and relative to each other. A low or plateauing FHR, impaired growth trajectory, discrepant sac development, or abnormal yolk sac pattern can alter risk interpretation, especially when combined with maternal age, BMI, previous pregnancy loss, parity, gravidity, and conception method. This creates a prediction problem that is time-sensitive, multimodal, and clinically interdependent.

The research framing in `Research.pdf` motivates two complementary approaches. PREG-Net is the primary model because it explicitly represents clinical variables and their relationships as a graph, supporting patient-specific node and edge explanations. FETA-Transformer is the temporal comparison model because it learns longitudinal patterns and gestational windows from repeated first-trimester scans. The combined research story is not simply to maximize AUROC; it is to compare temporal attention, graph attention, and late fusion as alternative ways of representing early pregnancy biology.

## Methods

### Cohort

We generated a synthetic first-trimester cohort using `synthetic_generator_v2.py` with seed 42. The final cohort contained 800 patients and 2408 ultrasound scan records. There were 612 ongoing pregnancies and 188 pregnancy losses, yielding a 23.5 percent loss rate. Each patient had 2-5 scans. Mean scans per patient was 3.01 +/- 0.82. The mean first scan gestational age was 6.50 +/- 0.67 weeks and the mean latest scan gestational age was 10.52 +/- 1.82 weeks.

Patient-level features included age, BMI, parity, gravidity, previous loss, conception method, singleton status, and outcome label. Ultrasound features were FHR, CRL, GS, and YSD. Synthetic trajectories were generated from biologically motivated first-trimester growth curves and patient-level latent factors. Pregnancy-loss trajectories had overlapping but shifted latent-factor distributions, higher noise, and potential longitudinal deterioration. Missingness was modeled as gestational-age and outcome dependent, then imputed within patient by interpolation, back-fill, and forward-fill with residual median imputation.

### Preprocessing

Patients were split into stratified train, validation, and test sets using a 70/15/15 ratio. Training-only normalization statistics were applied to continuous ultrasound features and maternal features. Sequences were padded or truncated to five scans, with temporal masks preserving which scan positions were real. Binary maternal features were left unnormalized.

### PREG-Net

PREG-Net converted each patient into a knowledge-guided graph. Temporal nodes represented FHR, CRL, GS, and YSD at each scan time point; static maternal nodes represented age, BMI, parity, gravidity, previous loss, conception method, and singleton status. Edges encoded physiological relationships, temporal continuity between consecutive same-variable nodes, and maternal-to-ultrasound influences. A pure PyTorch graph attention network performed message passing, followed by attention readout and binary classification. Node importance and edge attention were exported for interpretability.

### FETA-Transformer

FETA-Transformer processed longitudinal ultrasound measurements with modality-specific projections, continuous gestational-age positional encoding, Transformer encoder layers, temporal attention pooling, maternal conditioning, and a binary classification head. Temporal attention weights were exported by scan and summarized by gestational week.

### Ensembles

Two late-fusion ensembles combined PREG-Net and FETA-Transformer. The simple ensemble averaged logits. The learned ensemble trained a linear fusion layer over the two logits.

### Baselines

Baseline models used engineered tabular features: scan count, first and latest gestational age, gestational-age span, maternal features, and latest, mean, and gestational-age slope for each ultrasound variable. Baselines included logistic regression, MLP, Random Forest, HistGradientBoosting, XGBoost, and sklearn logistic regression.

### Statistical Analysis

Metrics included AUROC, AUPRC, accuracy, sensitivity, specificity, F1, precision, and recall. Bootstrap confidence intervals used 2000 resamples. Exact McNemar tests compared paired classification differences. Paired bootstrap AUROC-difference comparisons were used as a documented DeLong alternative. Calibration was assessed by Brier score, expected calibration error, maximum calibration error, and calibration curves.

## Results

### Fixed Test Split

PREG-Net achieved AUROC 0.9855 with 95 percent CI [0.9663, 0.9973], AUPRC 0.9568 [0.8909, 0.9914], accuracy 0.9508 [0.9098, 0.9836], sensitivity 0.9655 [0.8800, 1.0000], specificity 0.9462 [0.8942, 0.9886], and F1 0.9032 [0.8108, 0.9697].

FETA-Transformer achieved AUROC 0.9896 [0.9732, 0.9993] and AUPRC 0.9728 [0.9295, 0.9978]. The simple ensemble achieved AUROC 0.9933 [0.9810, 1.0000] and AUPRC 0.9806 [0.9408, 1.0000]. The learned ensemble achieved similar discrimination with AUROC 0.9933 [0.9824, 0.9997] and AUPRC 0.9807 [0.9469, 0.9991].

Tabular baselines performed strongly. The MLP achieved the highest fixed-split AUROC and AUPRC, with AUROC 0.9985 [0.9942, 1.0000] and AUPRC 0.9952 [0.9793, 1.0000]. This result should be interpreted as evidence that the synthetic generator produces strong tabular signal, and it tempers any claim that the deep models are superior in raw discrimination.

### Cross-Validation

Five-fold stratified cross-validation confirmed high discrimination across models. Sklearn logistic regression had the highest mean test AUROC at 0.9930 +/- 0.0042. Among research deep models, learned ensemble had mean AUROC 0.9873 +/- 0.0146 and mean F1 0.9306 +/- 0.0181. FETA-Transformer had mean AUROC 0.9859 +/- 0.0115. PREG-Net had mean AUROC 0.9761 +/- 0.0116.

### Explainability

PREG-Net node importance was highest for FHR, followed by YSD and selected maternal/context features. Edge attention highlighted both expected and additional implementation relationships, including CRL-to-GS, FHR-to-YSD, temporal same-variable edges, previous-loss-to-YSD, and FHR-to-CRL. FETA temporal attention emphasized later gestational weeks for many patients. Case studies were exported for true positive, true negative, false positive, and false negative examples.

## Discussion

The primary contribution of this work is PREG-Net, an interpretable graph attention framework for representing first-trimester clinical relationships. Its graph structure makes clinical assumptions explicit and supports node- and edge-level explanations. FETA-Transformer provides a complementary temporal baseline that models irregular gestational-age sequences and identifies attention-weighted scan windows. The ensemble results suggest that temporal and graph representations can be combined, although in this synthetic setting tabular baselines remained extremely competitive.

The finding that simple tabular models performed very strongly is important. It may reflect the synthetic generator's clear patient-level and trajectory-level signal, the relatively small number of scan time points, and the engineered features that capture latest values, means, and slopes. Therefore, the manuscript should not claim deep learning superiority. Instead, it should frame PREG-Net as an interpretable modeling contribution and FETA-Transformer as a temporal representation benchmark.

## Limitations

This study used synthetic data and cannot establish clinical validity. The simulator encodes assumptions about fetal growth, cardiac development, placental function, maternal risk, noise, and missingness; real-world data may deviate from these assumptions. External validation on independent clinical cohorts is required. Measurement variability, scan timing, operator practice, site-specific protocols, and outcome-label definitions may affect performance. The first-trimester sequences are short, limiting the benefit of complex temporal models. Maternal-to-time attention maps from FETA are post-hoc because the current architecture conditions on a pooled temporal token. Ablation experiments remain pending and should be completed before final manuscript claims about architectural components.

## Reproducibility

Key artifacts are saved under:

- `data/generated`: synthetic patients, scans, and dataset summary.
- `results/model_comparison.*`: fixed test split metrics.
- `results/cross_validation`: five-fold cross-validation summaries.
- `results/statistical_analysis`: confidence intervals, paired tests, and calibration.
- `results/explainability`: attention, graph importance, and case studies.
- `results/paper_artifacts`: figures and tables.

