# Implementation Plan: FETA-Transformer + PREG-Net

Last updated: 2026-07-04

## Research Direction From `Research.pdf`

The PDF proposes two complementary models for first-trimester pregnancy loss prediction:

- **Primary model:** PREG-Net, because it is biologically grounded, interpretable, and explicitly models relationships between clinical variables.
- **Secondary/comparison model:** FETA-Transformer, because it learns longitudinal ultrasound patterns and critical gestational windows.
- **Strongest research story:** train PREG-Net, FETA-Transformer, and an ensemble, then compare temporal attention, graph attention, and combined performance.

Target dataset shape from the PDF: 500-1000 patients, at least 2 ultrasound scans per patient, first-trimester longitudinal features `FHR`, `CRL`, `GS`, `YSD`, plus maternal features.

## Current Code Status

### Already Implemented

- `synthetic_generator_v2.py`
  - Generates 800 synthetic patients with about 22 percent pregnancy loss.
  - Generates 2-5 first-trimester scans per patient.
  - Includes biologically motivated curves, maternal risk features, missingness, and interpolation.

- `data/dataset.py`
  - Builds FETA-style temporal tensors:
    - `temporal_features`: `(N, T, 4)`
    - `gestational_ages`: `(N, T)`
    - `temporal_mask`: `(N, T)`
    - `maternal_features`: `(N, 7)`
    - `label`
  - Implements stratified train/validation/test splitting.
  - Computes train-only normalization stats.

- `data/graph_builder.py`
  - Builds PREG-Net graph inputs from a dataset batch.
  - Includes knowledge-guided edges, temporal edges, node types, node masks, and scalar node features.

- `models/feta_transformer.py`
  - Implements modality-specific projections, continuous positional encoding, Transformer encoder, attention pooling, maternal cross-attention, and classification head.

- `models/preg_net.py`
  - Implements a pure PyTorch GAT, graph readout, node importance, edge attention, and classification head.

- `models/ensemble.py`
  - Implements late fusion through `EnsemblePredictor`.
  - Supports average fusion and learned fusion.

- `utils/training.py`
  - Contains early stopping, weighted BCE loss, AdamW, cosine warm restarts, checkpointing, and train/validation loops.

- `utils/metrics.py`
  - Computes AUROC, AUPRC, accuracy, sensitivity, specificity, F1, precision, recall, and bootstrap confidence intervals.

### Current Blockers

1. `models/__init__.py` import blocker is fixed.
   - It now exports both `EnsemblePredictor` and the backward-compatible `EnsembleModel` alias.

2. `feta_preg.py` compile blocker is fixed.
   - The original Colab export is preserved at `notebooks/feta_preg_colab_export.txt`.
   - The top-level `feta_preg.py` is now a valid local launcher.

3. Synthetic dataset files now exist.
   - `data/generated/patients.csv`
   - `data/generated/scans.csv`
   - `data/generated/dataset_summary.json`
   - Patient JSON files, model checkpoints, and final result files are still pending.

4. The trainer is now wired to dataset batches for first-pass training.
   - `Trainer._unpack_batch()` accepts `label`.
   - `patient_id` is removed before model calls.
   - `collate_mode` now supports `feta`, `preg`, and `ensemble`.
   - PREG-Net and ensemble graph tensors are built with `PatientGraphBuilder`.

5. Ensemble/PREG masking is fixed for the forward path.
   - `EnsemblePredictor.forward()` accepts `node_mask`.
   - The trainer passes graph masks into PREG-Net and the ensemble.

6. Dependencies are now captured in `requirements.txt`.
   - `utils/metrics.py` also has a NumPy fallback for environments without `scikit-learn`.
   - PDF extraction tools were also missing globally, though this is not required for model training.

7. FETA positional encoding was corrected.
   - The research formula is `sin(t / theta^(2i/d))`.
   - The implementation now uses the equivalent `ga * theta^(-2i/d)` form.

8. FETA maternal cross-attention is not very informative yet.
   - It attends over a single pooled temporal token, so the attention distribution has no real temporal choice.
   - To support the PDF's interpretability claim, maternal features should attend over temporal tokens or modality-time tokens.

9. Missing research-grade experiment layer.
   - Data generation and smoke-test CLIs now exist.
   - PREG-Net, FETA-Transformer, ensemble, baseline, and cross-validation CLIs now exist.
   - Ablations, explainability, temporal baseline, and paper-ready figure-generation CLIs are still pending.
   - No saved split files or reproducibility manifest.
   - No tests.

## Step-by-Step TODO

### Step 1: Make The Repo Importable And Reproducible

- [x] Fix `models/__init__.py` to export `EnsemblePredictor`.
- [x] Decide what to do with `feta_preg.py`:
  - Convert it into a valid script, or
  - Move notebook-only code into `notebooks/`, or
  - Exclude it from compile/test commands.
- [x] Add `requirements.txt` with at least:
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- [x] Add a short `README.md` with setup, data generation, training, and evaluation commands.
- [x] Add a `scripts/` directory for runnable entry points.
- [x] Add a lightweight smoke test command that imports every package module.

### Step 2: Persist And Validate The Synthetic Dataset

- [x] Update `synthetic_generator_v2.py` so it can be called as a script:
  - `python synthetic_generator_v2.py --out data/generated --seed 42 --n-patients 800`
- [x] Save:
  - `data/generated/patients.csv`
  - `data/generated/scans.csv`
  - `data/generated/dataset_summary.json`
- [x] Validate the generated cohort:
  - 800 unique patients.
  - 2-5 scans per patient.
  - Loss rate near 22 percent.
  - No patient without scans.
  - First-trimester gestational ages only.
  - Required columns are present.
- [ ] Save plots for quick sanity checks:
  - CRL vs gestational age by outcome.
  - FHR vs gestational age by outcome.
  - Maternal age and BMI by outcome.
  - Missingness summary before and after imputation.

### Step 3: Finish Dataset Preparation

- [ ] Add split persistence:
  - [ ] `splits/train_ids.csv`
  - [ ] `splits/val_ids.csv`
  - [ ] `splits/test_ids.csv`
  - [x] CV `fold_0.json` through `fold_4.json` support in `results/cross_validation/splits/`
- [x] Add a 5-fold stratified cross-validation helper.
- [ ] Replace `nan_to_num(..., 0.0)` with the planned imputation pipeline:
  - per-patient forward fill
  - per-patient backward fill
  - train-set median imputation
- [ ] Store normalization parameters fitted on training data only.
- [x] Add optional patient-level tabular features for baselines:
  - latest value for each ultrasound variable
  - mean value
  - slope over gestational age
  - scan count
  - maternal features
- [ ] Add dataset unit tests for shapes, masks, labels, splits, and normalization leakage.

### Step 4: Repair Training Integration

- [x] Fix `Trainer._unpack_batch()` to read `label`, not `labels`.
- [x] Remove non-tensor metadata such as `patient_id` before calling model `forward()`.
- [x] Implement `collate_mode` properly:
  - `feta`: pass temporal and maternal tensors only.
  - `preg`: build graph tensors and pass graph inputs.
  - `ensemble`: pass both FETA tensors and graph tensors.
- [x] Add `PatientGraphBuilder` integration inside the trainer or inside a dedicated model-input adapter.
- [x] Pass `node_mask` into PREG-Net and the ensemble.
- [x] Add a single-batch overfit test for:
  - FETA-Transformer.
  - PREG-Net.
  - EnsemblePredictor.
- [x] Save training history as CSV and JSON.

### Step 5: Correct And Strengthen FETA-Transformer

- [x] Fix continuous positional encoding to match the PDF formula.
- [ ] Add tests for positional encoding shape and value sanity.
- [ ] Rework maternal cross-attention so maternal features attend over temporal tokens, not only the already pooled vector.
- [ ] Expose attention outputs needed for figures:
  - temporal pooling weights
  - transformer attention weights if practical
  - maternal-to-time attention weights
- [ ] Add ablation switches:
  - no continuous positional encoding
  - no attention pooling
  - no maternal cross-attention
- [x] Train and evaluate the full FETA-Transformer on fixed splits.
  - Command exists: `python3 scripts/train_feta.py --device cpu`
  - Full run completed in `results/feta_transformer`.
  - Test AUROC: 0.9896; test AUPRC: 0.9728; test F1: 0.8667.

### Step 6: Correct And Strengthen PREG-Net

- [ ] Confirm the graph edge list exactly matches the research rationale:
  - FHR to CRL
  - YSD to GS
  - Age to FHR/CRL
  - BMI to all ultrasound nodes
  - Prior loss to all ultrasound nodes
  - Temporal same-variable edges
- [ ] Document any additional edges already in code, such as CRL-GS, FHR-YSD, gravidity, parity, or conception edges.
- [ ] Ensure padded temporal nodes cannot receive or contribute meaningful messages.
- [ ] Save node importance and edge attention for each evaluated patient.
- [ ] Add ablation switches:
  - no knowledge edges
  - random graph
  - no temporal edges
  - no edge attention
- [x] Train and evaluate PREG-Net as the primary model.
  - First training command exists: `python3 scripts/train_preg.py --device cpu`
  - Full run completed in `results/preg_net`.
  - Test AUROC: 0.9855; test AUPRC: 0.9568; test F1: 0.9032.

### Step 7: Finalize Ensemble

- [x] Rename or alias classes consistently:
  - Either use `EnsemblePredictor` everywhere, or add an `EnsembleModel = EnsemblePredictor` alias.
- [ ] Decide whether average fusion should average logits or probabilities.
  - The original plan says probability averaging.
  - The current code averages logits.
- [x] Add `node_mask` to the ensemble forward pass.
- [x] Train/evaluate:
  - [x] simple average ensemble
  - [x] learned fusion ensemble
  - Command exists: `python3 scripts/train_ensemble.py --device cpu`
- [x] Compare ensemble performance against both individual models.
  - Comparison files: `results/model_comparison.csv`, `results/model_comparison.json`, `results/model_comparison.md`
  - Ensemble has best test AUROC/AUPRC.
  - PREG-Net has best test accuracy/specificity/F1.
  - Learned fusion gives almost identical test metrics to simple averaging.

### Step 8: Add Baselines

- [x] Implement tabular baselines on the same saved splits:
  - [x] Logistic Regression
  - [x] Random Forest
  - [x] HistGradientBoosting
  - [x] XGBoost
  - [x] MLP
  - Baseline script exists: `python3 scripts/train_baselines.py`
  - `scikit-learn` baselines completed after installing scikit-learn.
  - XGBoost completed after installing `xgboost`.
- [ ] Implement a temporal baseline:
  - LSTM or GRU over scan sequences
- [x] Save baseline predictions and metrics.

### Step 9: Evaluation And Statistical Rigor

- [x] Evaluate all models on the same test split.
- [x] Run 5-fold cross-validation for final reported metrics.
- [x] Report:
  - AUROC
  - AUPRC
  - sensitivity
  - specificity
  - F1
  - precision
  - accuracy
- [x] Compute 95 percent bootstrap confidence intervals.
- [x] Add McNemar's test for paired classification differences.
- [x] Add DeLong test or a documented AUROC comparison alternative.
- [x] Add calibration metrics and calibration plots.

### Step 10: Explainability Outputs

- [x] FETA explainability:
  - [x] temporal attention by gestational week
  - [x] maternal-to-temporal attention maps
  - [x] patient trajectory overlays
  - [ ] optional SHAP on learned embeddings or tabular baseline features
- [x] PREG-Net explainability:
  - [x] node importance ranking
  - [x] edge attention ranking
  - [x] patient-specific graph visualizations
  - [x] comparison of learned edge importance with clinical expectations
- [x] Select case studies:
  - [x] true positives
  - [x] true negatives
  - [x] false positives
  - [x] false negatives

### Step 11: Paper-Ready Figures And Tables

- [ ] Figures:
  - [x] FETA architecture
  - [x] PREG-Net architecture
  - [x] ROC curves
  - [x] precision-recall curves
  - [x] confusion matrices
  - [x] attention heatmaps
  - [x] graph importance visualizations
  - [x] patient trajectories
  - [ ] ablation bar charts
    - Status artifact generated only; real ablation runs are still pending.
  - [x] calibration plots
- [ ] Tables:
  - [x] dataset demographics by outcome
  - [x] model comparison metrics
  - [x] cross-validation mean and standard deviation
  - [x] confidence intervals
  - [ ] ablation results
    - Status table generated only; real ablation runs are still pending.
  - [x] hyperparameters
  - [x] computational cost

### Step 12: Manuscript And Synopsis Alignment

- [x] Position PREG-Net as the primary contribution.
- [x] Position FETA-Transformer as the temporal comparison model.
- [x] Use the combined justification from the PDF:
  - [x] early pregnancy loss is time-sensitive, multimodal, and clinically interdependent.
  - [x] temporal attention models gestational progression.
  - [x] graph attention models clinical relationships.
- [x] Write methods for:
  - [x] synthetic/clinical cohort construction
  - [x] preprocessing and split strategy
  - [x] PREG-Net
  - [x] FETA-Transformer
  - [x] ensemble
  - [x] baselines
  - [x] statistical analysis
- [x] Write limitations clearly:
  - [x] synthetic data limitations
  - [x] need for external validation
  - [x] possible measurement and missingness bias
  - [x] small first-trimester sequence length

## Recommended Immediate Next Tasks

1. Fix import blockers: complete.
   - `models/__init__.py`
   - `feta_preg.py` handling
   - dependency file

2. Make synthetic data generation persistent and reproducible: complete.
   - save `patients.csv`
   - save `scans.csv`
   - save dataset summary

3. Repair trainer-to-dataset integration: complete.
   - `label` key
   - remove `patient_id`
   - implement `feta`, `preg`, and `ensemble` input adapters

4. Run one-batch smoke training for all three model paths: complete.

5. Train PREG-Net first, because the PDF recommends it as the primary model.
   - Complete: `results/preg_net`

6. Train FETA-Transformer next for direct comparison.
   - Complete: `results/feta_transformer`

7. Train the ensemble next.
   - Complete: `results/ensemble`

8. Run learned-fusion ensemble next.
   - Complete: `results/ensemble_learned`

9. Add and run baseline models next.
   - Complete for Logistic Regression, MLP, Random Forest, HistGradientBoosting, and XGBoost: `results/baselines`

10. Add 5-fold cross-validation next.
   - Script exists: `python3 scripts/cross_validate.py`
   - Smoke test passed.
   - Full run complete:
     `results/cross_validation`

11. Add confidence intervals and statistical comparison tests next.
   - Complete: `results/statistical_analysis`
   - Includes bootstrap 95 percent CIs, McNemar tests, paired bootstrap AUROC differences, calibration metrics, and calibration curves.

12. Add explainability outputs next.
   - Complete: `results/explainability`
   - Includes FETA temporal and post-hoc maternal-to-time attention summaries, PREG-Net node and edge rankings, patient graph visualizations, trajectory overlays, and TP/TN/FP/FN case studies.
   - Optional SHAP remains deferred to avoid adding a new dependency for a non-required interpretability layer.

13. Add paper-ready figures and tables next.
   - Complete except real ablation results: `results/paper_artifacts`
   - Includes architecture diagrams, ROC/PR curves, confusion matrices, attention heatmaps, graph importance figures, patient trajectories, calibration curves, and dataset/model/CV/CI/hyperparameter/cost tables.
   - Ablation figure/table are status artifacts only because ablation experiments have not been implemented or run.

14. Implement and run ablations next.
   - Pending: add FETA and PREG-Net ablation switches, run ablation experiments, and regenerate `results/paper_artifacts`.

15. Align manuscript and synopsis next.
   - Complete: `results/manuscript`
   - Includes synopsis, manuscript draft, methods draft, limitations, and Step 12 alignment checklist.

## Verification Performed

- Read `implementation_plan.md`.
- Inspected project file tree.
- Extracted `Research.pdf` text with a temporary local PDF reader.
- Ran `python3 -m compileall .`.
  - Result: all Python files compile.
- Ran `python3 scripts/smoke_test.py`.
  - Result: FETA, PREG-Net, and ensemble forward paths work.
- Ran `python3 scripts/overfit_batch.py --mode all --steps 80 --cpu`.
  - Result: FETA, PREG-Net, and ensemble all reduced one-batch loss successfully.
- Ran `python3 scripts/train_preg.py --epochs 3 --batch-size 64 --patience 3 --out-dir results/preg_net_smoke --device cpu`.
  - Result: training, checkpointing, history export, prediction export, and metrics export all work.
  - Smoke test metrics are not final research metrics.
- Verified full PREG-Net run outputs in `results/preg_net`.
  - Test metrics: AUROC 0.9855, AUPRC 0.9568, accuracy 0.9508, sensitivity 0.9655, specificity 0.9462, F1 0.9032.
- Ran `python3 scripts/train_feta.py --epochs 3 --batch-size 64 --patience 3 --out-dir results/feta_transformer_smoke --device cpu`.
  - Result: FETA training, checkpointing, prediction export, attention export, and metrics export all work.
- Verified full FETA-Transformer run outputs in `results/feta_transformer`.
  - Test metrics: AUROC 0.9896, AUPRC 0.9728, accuracy 0.9344, sensitivity 0.8966, specificity 0.9462, F1 0.8667.
- Ran `python3 scripts/train_ensemble.py --epochs 3 --batch-size 64 --patience 3 --out-dir results/ensemble_smoke --device cpu`.
  - Result: ensemble training, checkpointing, prediction export, individual model probability export, and metrics export all work.
- Verified full ensemble run outputs in `results/ensemble`.
  - Test metrics: AUROC 0.9933, AUPRC 0.9806, accuracy 0.9426, sensitivity 0.9655, specificity 0.9355, F1 0.8889.
- Ran `python3 scripts/compare_models.py --split test`.
  - Result: saved comparison files in `results/model_comparison.csv`, `results/model_comparison.json`, and `results/model_comparison.md`.
- Verified full learned-fusion ensemble run outputs in `results/ensemble_learned`.
  - Test metrics: AUROC 0.9933, AUPRC 0.9807, accuracy 0.9426, sensitivity 0.9655, specificity 0.9355, F1 0.8889.
  - Learned fusion weights: FETA 0.5417, PREG 0.4810.
- Re-ran `python3 scripts/compare_models.py --split test`.
  - Result: comparison now includes PREG-Net, FETA-Transformer, simple ensemble, and learned ensemble.
- Ran `python3 scripts/train_baselines.py --epochs 300 --patience 50 --device cpu`.
  - Result: PyTorch logistic regression and MLP baselines completed.
  - Logistic regression test metrics: AUROC 0.9889, AUPRC 0.9661, accuracy 0.9016, sensitivity 0.9655, specificity 0.8817, F1 0.8235.
  - MLP test metrics: AUROC 0.9985, AUPRC 0.9952, accuracy 0.9836, sensitivity 1.0000, specificity 0.9785, F1 0.9667.
- Re-ran `python3 scripts/train_baselines.py --epochs 300 --patience 50 --device cpu` after installing scikit-learn.
  - Result: sklearn logistic regression, Random Forest, and HistGradientBoosting completed.
  - Sklearn logistic regression test metrics: AUROC 0.9963, AUPRC 0.9878, accuracy 0.9672, sensitivity 0.9655, specificity 0.9677, F1 0.9333.
  - Random Forest test metrics: AUROC 0.9909, AUPRC 0.9705, accuracy 0.9508, sensitivity 0.9310, specificity 0.9570, F1 0.9000.
  - HistGradientBoosting test metrics: AUROC 0.9933, AUPRC 0.9784, accuracy 0.9590, sensitivity 0.9310, specificity 0.9677, F1 0.9153.
- Re-ran `python3 scripts/train_baselines.py --epochs 300 --patience 50 --device cpu` after installing XGBoost.
  - Result: XGBoost completed.
  - XGBoost test metrics: AUROC 0.9911, AUPRC 0.9729, accuracy 0.9508, sensitivity 0.9655, specificity 0.9462, F1 0.9032.
- Re-ran `python3 scripts/compare_models.py --split test`.
  - Result: comparison now includes neural models plus Logistic Regression, MLP, Random Forest, HistGradientBoosting, and XGBoost baselines.
- Added `scripts/cross_validate.py`.
  - Supports stratified folds, train/validation/test fold splits, deep models, tabular baselines, fold metric CSV/JSON, summary CSV/JSON/Markdown, and saved fold IDs.
- Ran `python3 scripts/cross_validate.py --models preg,torch_logistic_regression --n-folds 2 --max-folds 1 --epochs 1 --patience 1 --batch-size 64 --out-dir results/cross_validation_smoke --device cpu`.
  - Result: CV smoke test passed and wrote outputs under `results/cross_validation_smoke`.
- Ran `python3 scripts/cross_validate.py --models preg,feta,ensemble,learned_ensemble,torch_mlp,sklearn_logistic_regression,random_forest,hist_gradient_boosting,xgboost --n-folds 5 --epochs 80 --batch-size 32 --out-dir results/cross_validation --device cpu`.
  - Result: full 5-fold CV completed and wrote `results/cross_validation/summary.md`.
  - Best mean test AUROC/AUPRC: sklearn logistic regression, AUROC 0.9930 +/- 0.0042, AUPRC 0.9806 +/- 0.0116.
  - Best mean test F1 among research models: learned ensemble, F1 0.9306 +/- 0.0181.
  - PREG-Net mean test AUROC: 0.9761 +/- 0.0116.
  - FETA mean test AUROC: 0.9859 +/- 0.0115.
  - Simple ensemble mean test AUROC: 0.9865 +/- 0.0133.
  - Learned ensemble mean test AUROC: 0.9873 +/- 0.0146.
- Checked trainer adapters on generated CSV data.
  - Result: `feta`, `preg`, and `ensemble` modes produce logits with expected shape.
- Generated persisted synthetic data.
  - Result: `data/generated/patients.csv`, `data/generated/scans.csv`, and `data/generated/dataset_summary.json` exist.
- Added `scripts/statistical_analysis.py`.
  - Supports bootstrap 95 percent CIs, exact McNemar tests, paired bootstrap AUROC-difference tests, Brier score, ECE/MCE, and calibration-curve output.
- Ran `python3 scripts/statistical_analysis.py --iterations 2000 --out-dir results/statistical_analysis`.
  - Result: statistical outputs saved under `results/statistical_analysis`.
  - Top AUROC bootstrap comparison: PREG-Net vs MLP, AUROC difference -0.0130, 95 percent CI [-0.0297, -0.0023], p=0.0120.
  - Best Brier score on the fixed test split: MLP, 0.0162.
- Added `scripts/export_explainability.py`.
  - Supports FETA temporal attention by scan/week, post-hoc maternal-to-time attention maps, PREG-Net node importance, PREG-Net edge attention, clinical edge comparison, and case-study SVGs.
- Ran `python3 scripts/export_explainability.py --out-dir results/explainability --device cpu`.
  - Result: explainability outputs saved under `results/explainability`.
  - FETA scan attention rows: 359.
  - PREG-Net node importance rows: 2290.
  - PREG-Net edge attention rows: 11589.
  - Selected case-study patients: P00548 true positive, P00320 true negative, P00089 false positive, P00099 false negative.
- Added `scripts/generate_paper_artifacts.py`.
  - Supports paper-ready SVG figures and CSV/Markdown/LaTeX table bundles.
  - Sources saved model predictions, cross-validation summaries, statistical analysis outputs, and explainability exports.
- Ran `python3 scripts/generate_paper_artifacts.py --out-dir results/paper_artifacts`.
  - Result: 19 SVG figures and 21 table files saved under `results/paper_artifacts`.
  - Includes ablation status artifacts only; no real ablation result files exist yet.
- Added manuscript alignment drafts under `results/manuscript`.
  - Files: `synopsis.md`, `manuscript_draft.md`, `methods.md`, `limitations.md`, and `alignment_checklist.md`.
  - Result: Step 12 positioning, methods, justification, and limitations are covered.
