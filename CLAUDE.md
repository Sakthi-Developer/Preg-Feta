# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for **first-trimester pregnancy-loss prediction** implementing the two models proposed in `Research.pdf`:

- **PREG-Net** — a knowledge-guided Graph Attention Network over clinical variables (`models/preg_net.py`).
- **FETA-Transformer** — a temporal attention Transformer over longitudinal ultrasound scans (`models/feta_transformer.py`).
- **EnsemblePredictor** — late fusion of the two, either average or learned (`models/ensemble.py`).

There is **no real patient data**; all experiments run on synthetic data produced by `synthetic_generator_v2.py`. The end goal is a publishable manuscript (drafts live in `results/manuscript/`), so scripts emit paper-ready artifacts (CSV/Markdown/LaTeX tables, SVG figures, bootstrap CIs, explainability exports).

Both models are **pure PyTorch — no PyTorch Geometric.** The GAT message passing in `preg_net.py` is implemented by hand with `scatter_add_`/sparse softmax. Preserve this; do not introduce `torch_geometric`.

## Environment & how to run

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # torch, numpy, pandas, scikit-learn, matplotlib, seaborn
```

**Always run scripts from the repository root.** This is not an installable package — every entry point does `sys.path.insert(0, ROOT)` and then uses root-anchored absolute imports (`from data.dataset import ...`, `from models import ...`). Running a script from inside `scripts/` will break imports.

`xgboost` is an **optional** baseline: it is imported behind a `try/except` and is *not* in `requirements.txt`. Baseline/CV code skips it gracefully when missing.

### Typical workflow (in order)

```bash
# 1. Generate synthetic cohort (writes data/generated/{patients,scans}.csv + dataset_summary.json)
python synthetic_generator_v2.py --out data/generated --seed 42 --n-patients 800

# 2. Wiring smoke test (forward passes on random tensors, no data needed)
python scripts/smoke_test.py

# 3. Sanity: can each model overfit a single real batch?
python scripts/overfit_batch.py --mode all      # or --mode feta|preg|ensemble

# 4. Train individual models (each writes to results/<model>/)
python scripts/train_feta.py --epochs 80
python scripts/train_preg.py --epochs 80
python scripts/train_ensemble.py --epochs 80
python scripts/train_baselines.py                # tabular baselines (LR, RF, HGB, MLP, xgboost)

# 5. Evaluation & paper artifacts
python scripts/compare_models.py --split test
python scripts/cross_validate.py --models preg,feta,ensemble,learned_ensemble,torch_mlp,sklearn_logistic_regression,random_forest,hist_gradient_boosting,xgboost --n-folds 5 --epochs 80 --device cpu --out-dir results/cross_validation
python scripts/statistical_analysis.py --iterations 2000 --out-dir results/statistical_analysis
python scripts/export_explainability.py --device cpu --out-dir results/explainability
python scripts/generate_paper_artifacts.py --out-dir results/paper_artifacts
```

There is **no test runner** (no pytest/unittest). `scripts/smoke_test.py` and `scripts/overfit_batch.py` are the closest thing to a test suite — run them after touching model or data code. `overfit_batch.py` exits non-zero if a model can't reduce loss on one batch, so it doubles as a regression check.

### Device convention

Training/eval scripts default to `--device cpu` deliberately (widest Mac compatibility). `utils.training.get_best_device()` auto-detects `cuda > mps > cpu` and is used when `--device auto` is passed. Keep CPU as the default for reproducibility.

## Architecture: how the pieces connect

The single most important thing to understand is how **one dataset feeds three different model signatures**.

### Data flow

1. `synthetic_generator_v2.py` → `data/generated/patients.csv` (one row/patient: maternal features + `label`) and `scans.csv` (many rows/patient: `FHR, CRL, GS, YSD` at a `gestational_age_weeks`).
2. `data/dataset.py::get_dataloaders()` loads both CSVs, does a **stratified** split (`create_splits`), computes **z-score normalization stats on the training split only** (`compute_norm_stats` — binary maternal features are intentionally *not* normalized), and returns `{train,val,test}` `DataLoader`s. Each item is padded/truncated to `T = max_scans = 5` and yields:
   - `temporal_features (T,4)`, `gestational_ages (T,)`, `temporal_mask (T,)`, `maternal_features (7,)`, `label`.
3. `data/graph_builder.py::PatientGraphBuilder.batch_to_graph()` converts a FETA-style batch into PREG-Net graph inputs **on the fly** (`node_features (B,N,1)`, shared `edge_index (2,E)`, `node_types (N,)`, `node_mask (B,N)`). The graph is **knowledge-guided and hardcoded** — physiological edges (e.g. FHR↔CRL), temporal edges (same variable across time), and maternal→US edges are defined by constants in `graph_builder.py`, not learned. Node count `N = max_scans*4 + 7 = 27`; there are 11 node *types*.

So PREG-Net never sees a separate data file — its graph is *derived* from the same temporal/maternal tensors.

### The `collate_mode` abstraction (critical)

`utils/training.py::Trainer` trains all three model families with one loop. `collate_mode ∈ {"feta","preg","ensemble"}` controls how `_unpack_batch` assembles kwargs and therefore which forward signature is called:

- `"feta"` → passes temporal + maternal tensors only.
- `"preg"` → runs `PatientGraphBuilder` and passes graph tensors only.
- `"ensemble"` → passes **both** sets of tensors.

`Trainer` tolerates both tuple outputs (`FETATransformer`, `PREGNet` return `(logits, attention_dict)`) and dict outputs (`EnsemblePredictor` returns a dict with `logits`, per-model logits, and attention). When adding a model, wire it through `collate_mode` rather than writing a new loop. `cross_validate.py::build_deep_model()` and each `train_*.py` set the matching mode (`learned_ensemble` = `EnsemblePredictor(learned_fusion=True)` with `collate_mode="ensemble"`).

### Class imbalance

Cohort is ~78% viable / ~22% loss. Loss is `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` via `compute_class_weights`. Models output **raw logits**; apply `sigmoid` for probabilities. Metric of record for checkpointing/early-stopping is **validation AUROC** (`mode="max"`).

### Explainability is a first-class output, not an afterthought

Every model returns attention weights and the training/eval scripts persist them (e.g. `train_feta.py` writes per-patient `temporal_attention_t1..t5` columns into `*_predictions.csv`). FETA exposes temporal-pooling + maternal-cross-attention weights; PREG-Net exposes node importance + per-layer edge attention. `export_explainability.py` turns these into figures and case studies. When modifying a model's forward pass, keep the attention dict keys stable — downstream exporters depend on them.

## Conventions & gotchas

- **`configs/default.py` is a reference, not wired in.** It centralizes hyperparameters as dataclasses (`get_default_config()`), but **no script imports it** — each script defines its own argparse defaults, and some diverge (e.g. `TrainingConfig.epochs=200` vs. `train_feta.py --epochs 80`). Treat CLI args as the source of truth for a given run; if you make the configs authoritative, update it consciously across all scripts.
- **Not a git repository** and has no CI. There is no lint/format config; match the existing style (module docstrings with `─` rule banners, `from __future__ import annotations`, full type hints, NumPy-style docstrings).
- **`results/`** holds committed experiment outputs; `*_smoke/` variants are throwaway quick runs. `tmp/` and `data/generated/` are regenerable. `results/manuscript/` holds the human-written paper drafts (`synopsis.md`, `manuscript_draft.md`, `methods.md`, `final_paper.md`, `limitations.md`, `alignment_checklist.md`).
- **`utils/metrics.py` ships pure-NumPy fallbacks** for AUROC/AUPRC/etc. behind a `try: from sklearn ...` guard, so metrics work even without scikit-learn. `compute_all_metrics` returns `nan` for AUROC/AUPRC/specificity when `y_true` has a single class — bootstrap/CV code skips degenerate resamples rather than crashing.
- `feta_preg.py` is a thin convenience launcher for data generation; `notebooks/feta_preg_colab_export.txt` is the original Colab export the project grew out of.

## Current research priority

Per `README.md` and `implementation_plan.md`: implement and run the **FETA/PREG-Net ablation experiments**, then regenerate paper artifacts with real ablation numbers. The ablation files currently emitted by `generate_paper_artifacts.py` are **status placeholders, not real results** — do not treat them as measured.
