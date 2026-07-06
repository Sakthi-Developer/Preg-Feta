# Limitations

## Synthetic Data

All experiments were performed on a synthetic cohort. The generator encodes assumptions about fetal growth, heart-rate trajectories, gestational sac growth, yolk sac behavior, maternal risk factors, measurement noise, missingness, and outcome-dependent deterioration. These assumptions make the dataset useful for controlled model development, but they also mean that performance estimates cannot be interpreted as clinical performance.

## Need For External Validation

No external clinical validation cohort was used. Before any deployment-oriented or clinical claim, the models should be evaluated on real first-trimester ultrasound cohorts from independent sites. Multi-center validation would be especially important because ultrasound measurement practice, scan timing, patient risk distribution, and pregnancy-loss definitions can vary across settings.

## Measurement And Missingness Bias

The generator models plausible measurement error and gestational-age-dependent missingness, but real missingness is often influenced by clinical workflow, image quality, patient presentation, prior risk, operator behavior, and documentation practice. These mechanisms may be informative and may differ from synthetic assumptions. The current saved dataset has missing ultrasound values imputed before modeling, which reduces complexity relative to real clinical data.

## Short First-Trimester Sequences

Each patient has only 2-5 scans. This is realistic for many first-trimester settings, but it limits the temporal depth available to Transformer models. Engineered tabular features such as latest value, mean, and slope can capture much of the signal in short sequences, which likely contributed to the strong baseline performance.

## Synthetic Signal And Baseline Strength

Tabular baselines, especially MLP and sklearn logistic regression, performed very strongly. This suggests that the synthetic generator produces structured tabular signal that can be captured without complex temporal or graph models. The manuscript should therefore avoid claiming that PREG-Net or FETA-Transformer outperform all baselines. PREG-Net should be framed as the primary interpretable architecture, not as the universally best discriminator in the current synthetic benchmark.

## Interpretability Limits

PREG-Net node importance and edge attention provide useful patient-specific explanations, but attention weights are not causal evidence. High attention does not prove that a variable caused the prediction or the outcome. FETA temporal attention weights identify influential scans, but the current maternal cross-attention layer attends over a pooled temporal token. The exported maternal-to-time maps are post-hoc interpretability aids derived from trained projections, not a fully retrained maternal-to-token attention mechanism.

## Pending Ablations

Ablation switches and experiments are not yet implemented or run. Final claims about the necessity of continuous positional encoding, attention pooling, maternal cross-attention, graph knowledge edges, temporal graph edges, or edge attention should wait until ablation experiments are complete.

