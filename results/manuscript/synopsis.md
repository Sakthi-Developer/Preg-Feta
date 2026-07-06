# Synopsis

## Working Title

Knowledge-Guided Graph Attention and Temporal Transformer Models for First-Trimester Pregnancy Loss Prediction in a Synthetic Longitudinal Ultrasound Cohort

## Central Claim

This study evaluates two complementary deep learning strategies for early pregnancy loss prediction from first-trimester longitudinal ultrasound and maternal features. PREG-Net is positioned as the primary methodological contribution because it represents clinical measurements as an interpretable knowledge-guided graph. FETA-Transformer is positioned as the temporal comparison model because it directly learns gestational-age-dependent longitudinal patterns from repeated ultrasound scans.

## Rationale

Early pregnancy loss prediction is time-sensitive, multimodal, and clinically interdependent. Fetal heart rate, crown-rump length, gestational sac diameter, yolk sac diameter, and maternal risk factors do not act independently; their interpretation depends on gestational age and on relationships among developmental, placental, and maternal variables. This motivates a dual-model design:

- PREG-Net models clinical interdependencies through knowledge-guided graph attention.
- FETA-Transformer models gestational progression through temporal attention over first-trimester scan sequences.
- A late-fusion ensemble tests whether graph-based and temporal representations provide complementary signal.

## Data And Evaluation

Experiments used a synthetic cohort of 800 patients and 2408 scans generated from biologically motivated first-trimester curves and maternal risk factors. The generated cohort contained 188 pregnancy losses and 612 ongoing pregnancies, for a 23.5 percent loss rate. Each patient had 2-5 scans with first-trimester ultrasound features FHR, CRL, GS, and YSD plus maternal features age, BMI, parity, gravidity, previous loss, conception method, and singleton status.

Models were evaluated on a fixed stratified train/validation/test split and with 5-fold stratified cross-validation. Primary metrics were AUROC, AUPRC, accuracy, sensitivity, specificity, F1, and precision. Statistical analysis included bootstrap 95 percent confidence intervals, exact McNemar tests for paired classification differences, paired bootstrap AUROC-difference comparisons, and calibration metrics.

## Main Results To Report

On the fixed test split, PREG-Net reached AUROC 0.9855, AUPRC 0.9568, accuracy 0.9508, sensitivity 0.9655, specificity 0.9462, and F1 0.9032. FETA-Transformer reached AUROC 0.9896 and AUPRC 0.9728. The simple and learned ensembles reached AUROC 0.9933 and AUPRC approximately 0.981. Tabular baselines were also strong, with the MLP achieving the highest fixed-split AUROC and AUPRC. This should be reported transparently: PREG-Net is the primary interpretable architecture, not the single best raw-discrimination model in every analysis.

In 5-fold cross-validation, sklearn logistic regression had the highest mean test AUROC at 0.9930 +/- 0.0042. Among the research deep models, learned ensemble had the strongest mean F1 at 0.9306 +/- 0.0181, while FETA and ensemble variants had higher mean AUROC than PREG-Net.

## Interpretability Findings

FETA temporal attention concentrated on later gestational windows in many patients. PREG-Net node importance was highest for FHR, followed by YSD and selected maternal/context nodes. Edge attention highlighted both predefined and additional implementation edges, including CRL-to-GS, FHR-to-YSD, temporal same-variable edges, previous-loss-to-YSD, and FHR-to-CRL. Case studies were selected for true positive, true negative, false positive, and false negative examples.

## Limitations

This work is a synthetic-cohort proof of concept, not a validated clinical model. External validation on real multi-center first-trimester ultrasound cohorts is required before any clinical claim. Measurement error, missingness, label definition, scan timing, and site-specific ultrasound practice may differ from the simulator. First-trimester sequences are short, which limits temporal modeling depth. FETA maternal-to-time attention maps are post-hoc interpretability aids because the current trained architecture applies maternal cross-attention to a pooled temporal token. Ablation experiments remain pending.

