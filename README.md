# AVM Pregnancy Loss Prediction

This repo implements the two research models described in `Research.pdf`:

- PREG-Net: a knowledge-guided graph attention model for clinical relationships.
- FETA-Transformer: a temporal attention model for longitudinal ultrasound scans.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Generate Synthetic Data

```bash
python synthetic_generator_v2.py --out data/generated --seed 42 --n-patients 800
```

This writes:

- `data/generated/patients.csv`
- `data/generated/scans.csv`
- `data/generated/dataset_summary.json`

## Smoke Test

```bash
python scripts/smoke_test.py
```

The smoke test runs forward passes through FETA-Transformer, PREG-Net, and the
ensemble on synthetic tensors.

## Evaluation

```bash
python scripts/compare_models.py --split test
python scripts/cross_validate.py --models preg,feta,ensemble,learned_ensemble,torch_mlp,sklearn_logistic_regression,random_forest,hist_gradient_boosting,xgboost --n-folds 5 --epochs 80 --batch-size 32 --out-dir results/cross_validation --device cpu
python scripts/statistical_analysis.py --iterations 2000 --out-dir results/statistical_analysis
python scripts/export_explainability.py --out-dir results/explainability --device cpu
python scripts/generate_paper_artifacts.py --out-dir results/paper_artifacts
```

The statistical analysis command writes bootstrap confidence intervals,
McNemar tests, paired bootstrap AUROC comparisons, calibration metrics, and
calibration curves.

The explainability export writes FETA temporal attention summaries, post-hoc
maternal-to-time maps, PREG-Net node and edge rankings, patient graph SVGs,
trajectory overlays, and TP/TN/FP/FN case studies.

The paper artifact command writes SVG figures plus CSV, Markdown, and LaTeX
tables under `results/paper_artifacts`. Real ablation results are still pending;
the generated ablation files are status artifacts only.

## Manuscript Drafts

Manuscript-facing drafts are under `results/manuscript`:

- `synopsis.md`
- `manuscript_draft.md`
- `methods.md`
- `limitations.md`
- `alignment_checklist.md`

## Current Priority

The next research milestone is to implement and run the FETA/PREG-Net ablation
experiments, then regenerate the paper artifacts with real ablation results.
