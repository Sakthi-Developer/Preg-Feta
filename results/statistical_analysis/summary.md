# Statistical Analysis Summary

- Models analyzed: 10
- Test patients: 122
- Bootstrap iterations requested: 2000
- AUROC comparison method: paired bootstrap difference, used as the documented DeLong alternative.

## Bootstrap Confidence Intervals

| Model | Metric | Estimate | 95% CI |
| --- | --- | --- | --- |
| PREG-Net | auroc | 0.9855 | [0.9663, 0.9973] |
| PREG-Net | auprc | 0.9568 | [0.8909, 0.9914] |
| PREG-Net | accuracy | 0.9508 | [0.9098, 0.9836] |
| PREG-Net | sensitivity | 0.9655 | [0.8800, 1.0000] |
| PREG-Net | specificity | 0.9462 | [0.8942, 0.9886] |
| PREG-Net | f1 | 0.9032 | [0.8108, 0.9697] |
| PREG-Net | precision | 0.8485 | [0.7143, 0.9630] |
| FETA-Transformer | auroc | 0.9896 | [0.9732, 0.9993] |
| FETA-Transformer | auprc | 0.9728 | [0.9295, 0.9978] |
| FETA-Transformer | accuracy | 0.9344 | [0.8852, 0.9754] |
| FETA-Transformer | sensitivity | 0.8966 | [0.7742, 1.0000] |
| FETA-Transformer | specificity | 0.9462 | [0.8961, 0.9890] |
| FETA-Transformer | f1 | 0.8667 | [0.7556, 0.9492] |
| FETA-Transformer | precision | 0.8387 | [0.6944, 0.9655] |
| Ensemble | auroc | 0.9933 | [0.9810, 1.0000] |
| Ensemble | auprc | 0.9806 | [0.9408, 1.0000] |
| Ensemble | accuracy | 0.9426 | [0.8934, 0.9836] |
| Ensemble | sensitivity | 0.9655 | [0.8800, 1.0000] |
| Ensemble | specificity | 0.9355 | [0.8788, 0.9789] |
| Ensemble | f1 | 0.8889 | [0.7937, 0.9620] |
| Ensemble | precision | 0.8235 | [0.6875, 0.9412] |
| Learned Ensemble | auroc | 0.9933 | [0.9824, 0.9997] |
| Learned Ensemble | auprc | 0.9807 | [0.9469, 0.9991] |
| Learned Ensemble | accuracy | 0.9426 | [0.9016, 0.9836] |
| Learned Ensemble | sensitivity | 0.9655 | [0.8800, 1.0000] |
| Learned Ensemble | specificity | 0.9355 | [0.8830, 0.9794] |
| Learned Ensemble | f1 | 0.8889 | [0.7931, 0.9630] |
| Learned Ensemble | precision | 0.8235 | [0.6875, 0.9444] |
| Torch Logistic Regression | auroc | 0.9889 | [0.9733, 0.9992] |
| Torch Logistic Regression | auprc | 0.9661 | [0.9117, 0.9977] |
| Torch Logistic Regression | accuracy | 0.9016 | [0.8443, 0.9508] |
| Torch Logistic Regression | sensitivity | 0.9655 | [0.8846, 1.0000] |
| Torch Logistic Regression | specificity | 0.8817 | [0.8119, 0.9432] |
| Torch Logistic Regression | f1 | 0.8235 | [0.7119, 0.9111] |
| Torch Logistic Regression | precision | 0.7179 | [0.5713, 0.8529] |
| MLP | auroc | 0.9985 | [0.9942, 1.0000] |
| MLP | auprc | 0.9952 | [0.9793, 1.0000] |
| MLP | accuracy | 0.9836 | [0.9590, 1.0000] |
| MLP | sensitivity | 1.0000 | [1.0000, 1.0000] |
| MLP | specificity | 0.9785 | [0.9468, 1.0000] |
| MLP | f1 | 0.9667 | [0.9123, 1.0000] |
| MLP | precision | 0.9355 | [0.8387, 1.0000] |
| Sklearn Logistic Regression | auroc | 0.9963 | [0.9881, 1.0000] |
| Sklearn Logistic Regression | auprc | 0.9878 | [0.9592, 1.0000] |
| Sklearn Logistic Regression | accuracy | 0.9672 | [0.9344, 0.9918] |
| Sklearn Logistic Regression | sensitivity | 0.9655 | [0.8800, 1.0000] |
| Sklearn Logistic Regression | specificity | 0.9677 | [0.9278, 1.0000] |
| Sklearn Logistic Regression | f1 | 0.9333 | [0.8571, 0.9855] |
| Sklearn Logistic Regression | precision | 0.9032 | [0.7931, 1.0000] |
| Random Forest | auroc | 0.9909 | [0.9774, 0.9992] |
| Random Forest | auprc | 0.9705 | [0.9239, 0.9974] |
| Random Forest | accuracy | 0.9508 | [0.9098, 0.9836] |
| Random Forest | sensitivity | 0.9310 | [0.8261, 1.0000] |
| Random Forest | specificity | 0.9570 | [0.9121, 0.9897] |
| Random Forest | f1 | 0.9000 | [0.8077, 0.9688] |
| Random Forest | precision | 0.8710 | [0.7391, 0.9697] |
| HistGradientBoosting | auroc | 0.9933 | [0.9814, 1.0000] |
| HistGradientBoosting | auprc | 0.9784 | [0.9393, 1.0000] |
| HistGradientBoosting | accuracy | 0.9590 | [0.9180, 0.9918] |
| HistGradientBoosting | sensitivity | 0.9310 | [0.8182, 1.0000] |
| HistGradientBoosting | specificity | 0.9677 | [0.9277, 1.0000] |
| HistGradientBoosting | f1 | 0.9153 | [0.8276, 0.9818] |
| HistGradientBoosting | precision | 0.9000 | [0.7838, 1.0000] |
| XGBoost | auroc | 0.9911 | [0.9768, 0.9992] |
| XGBoost | auprc | 0.9729 | [0.9280, 0.9973] |
| XGBoost | accuracy | 0.9508 | [0.9098, 0.9836] |
| XGBoost | sensitivity | 0.9655 | [0.8823, 1.0000] |
| XGBoost | specificity | 0.9462 | [0.8958, 0.9892] |
| XGBoost | f1 | 0.9032 | [0.8136, 0.9714] |
| XGBoost | precision | 0.8485 | [0.7143, 0.9677] |

## Smallest McNemar P Values

| Model A | Model B | Discordant | Exact p |
| --- | --- | ---: | ---: |
| Torch Logistic Regression | MLP | 10 | 0.0020 |
| Torch Logistic Regression | Sklearn Logistic Regression | 10 | 0.0215 |
| FETA-Transformer | MLP | 6 | 0.0312 |
| Ensemble | MLP | 5 | 0.0625 |
| Learned Ensemble | MLP | 5 | 0.0625 |
| Torch Logistic Regression | HistGradientBoosting | 13 | 0.0923 |
| PREG-Net | MLP | 4 | 0.1250 |
| MLP | Random Forest | 4 | 0.1250 |
| MLP | XGBoost | 4 | 0.1250 |
| PREG-Net | Torch Logistic Regression | 12 | 0.1460 |

## Smallest AUROC-Difference P Values

| Model A | Model B | AUROC diff | 95% CI | Bootstrap p |
| --- | --- | ---: | --- | ---: |
| PREG-Net | MLP | -0.0130 | [-0.0297, -0.0023] | 0.0120 |
| FETA-Transformer | MLP | -0.0089 | [-0.0232, -0.0007] | 0.0290 |
| PREG-Net | Sklearn Logistic Regression | -0.0108 | [-0.0273, -0.0007] | 0.0370 |
| MLP | XGBoost | 0.0074 | [0.0004, 0.0194] | 0.0430 |
| MLP | HistGradientBoosting | 0.0052 | [0.0000, 0.0145] | 0.0660 |
| Torch Logistic Regression | MLP | -0.0096 | [-0.0260, 0.0000] | 0.0700 |
| MLP | Random Forest | 0.0076 | [-0.0004, 0.0210] | 0.1010 |
| PREG-Net | Learned Ensemble | -0.0078 | [-0.0209, 0.0015] | 0.1330 |
| FETA-Transformer | Sklearn Logistic Regression | -0.0067 | [-0.0198, 0.0017] | 0.1640 |
| Sklearn Logistic Regression | Random Forest | 0.0054 | [-0.0013, 0.0169] | 0.1730 |

## Calibration Metrics

| Model | Brier score | ECE | MCE | Mean prediction | Prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.0162 | 0.0200 | 0.4438 | 0.2577 | 0.2377 |
| Sklearn Logistic Regression | 0.0209 | 0.0327 | 0.5579 | 0.2605 | 0.2377 |
| XGBoost | 0.0325 | 0.0245 | 0.8911 | 0.2521 | 0.2377 |
| HistGradientBoosting | 0.0347 | 0.0388 | 0.8491 | 0.2442 | 0.2377 |
| Random Forest | 0.0400 | 0.0599 | 0.2405 | 0.2582 | 0.2377 |
| Ensemble | 0.0426 | 0.0416 | 0.6356 | 0.2793 | 0.2377 |
| FETA-Transformer | 0.0454 | 0.0471 | 0.7667 | 0.2426 | 0.2377 |
| Learned Ensemble | 0.0466 | 0.0644 | 0.8827 | 0.2935 | 0.2377 |
| PREG-Net | 0.0502 | 0.0541 | 0.4465 | 0.2839 | 0.2377 |
| Torch Logistic Regression | 0.0999 | 0.2268 | 0.4393 | 0.4195 | 0.2377 |
