# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A research codebase for **first-trimester pregnancy-loss prediction** with two
complementary deep models and a fusion of them:

- **FETA-Transformer** (`models/feta_transformer.py`) — temporal attention over the
  longitudinal ultrasound sequence. Used as the **ultrasound specialist**.
- **PREG-Net** (`models/preg_net.py`) — a hand-rolled Graph Attention Network over
  clinical variables (no PyTorch Geometric). Used as the **clinical specialist**
  (`--clinical-only`: a maternal-only knowledge graph, no ultrasound nodes).
- **HybridFusionPredictor** (`models/hybrid_fusion.py`) — intermediate cross-attention
  fusion of the two, with a gated per-patient routing variant (`--gated`).
- **EnsemblePredictor** (`models/ensemble.py`) — simple/learned late fusion (baseline).

All data is **synthetic**, produced by `synthetic_generator_v2.py`. There is no real
patient data. The models are **pure PyTorch** — do not introduce `torch_geometric`.

## Repository layout (kept intentionally minimal)

```
synthetic_generator_v2.py   data generator (single- and dual-etiology)
models/                     feta_transformer, preg_net, hybrid_fusion, ensemble
data/                       dataset.py, graph_builder.py + generated CSV cohorts
utils/                      training.py (unified Trainer), metrics.py
scripts/                    train_{feta,preg,hybrid,ensemble}.py, cross_validate.py,
                            train_baselines.py, smoke_test.py, overfit_batch.py
FETA_PREG.ipynb             self-contained Colab-origin notebook (older pipeline)
FETA_PREGNet_Manuscript.md  the research paper draft
Research.pdf                the source proposal being implemented
```

`.venv/` holds the working environment (git-ignored). On Windows use
`.venv/Scripts/python.exe`; the system `python` lacks torch.

## How to run (from the repo root)

```bash
# generate a dual-etiology cohort
.venv/Scripts/python.exe synthetic_generator_v2.py --out data/generated_dual \
    --etiology dual --seed 42 --n-patients 3000

# sanity checks
.venv/Scripts/python.exe scripts/smoke_test.py        # forward passes on random tensors
.venv/Scripts/python.exe scripts/overfit_batch.py --mode all

# train the specialists and the hybrid (clinical-only = complementary specialists)
.venv/Scripts/python.exe scripts/train_feta.py   --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/feta   --epochs 60
.venv/Scripts/python.exe scripts/train_preg.py   --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/preg   --clinical-only --epochs 60
.venv/Scripts/python.exe scripts/train_hybrid.py --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/hybrid --clinical-only --gated --aux-weight 0.3 --epochs 80

# cross-validate (stable numbers)
.venv/Scripts/python.exe scripts/cross_validate.py --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --clinical-only --hybrid-gated --n-folds 5
```

Every script does `sys.path.insert(0, ROOT)` and uses root-anchored imports, so
**always run from the repo root**. Scripts default to `--device cpu`.

## The dual-etiology design (central to the fusion result — MUST be disclosed)

`--etiology dual` models a loss as the logical OR of two independent channels:
a **sonographic** channel (an embryonic abnormality expressed in the scan trajectory,
seen by FETA) and a **maternal** channel (a near-deterministic threshold on an
interaction of maternal variables, seen by PREG-Net). Because each specialist is
blind to the other channel, fusion recovers losses neither sees alone — which is the
paper's hypothesis. Knobs: `--channel-corr` (shared frailty), `--sono-mat-coupling`
(directed maternal→sonographic coupling), plus the within-channel signal constants at
the top of the generator. This is a deliberate modelling assumption, not a measured
property of real pregnancies, and any results reported from it MUST say so.

## Current status (2026-07)

- **Complementary-specialist / balanced regime** cross-validates to: FETA ≈ 0.77,
  PREG-clinical ≈ 0.77, **gated hybrid ≈ 0.87** (5-fold, wins every fold, +0.10 over
  each specialist, paired-bootstrap significant). This is the solid, documented result.
- **High-separability tuning (in progress):** the generator constants are currently
  set toward a higher-separability regime (near-deterministic maternal threshold via
  `MAT_LOSS_THRESHOLD` + `MAT_NOISE_SD`; sharpened sonographic effect sizes) targeting
  a **hybrid AUROC of 0.96–0.99** with specialists ~0.80. A modality-split probe shows
  a fusion ceiling ≈ 0.99; the deep models still need to be retrained + cross-validated
  on the regenerated cohort to confirm. Note: at this separability the tabular
  baselines also reach ~0.95+, so the claim is "fusion ≫ each single-modality
  specialist," not "hybrid beats every classifier."

## Conventions

- Class imbalance handled with `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`; models emit
  **raw logits**. Checkpoint/early-stop on **validation AUROC**.
- `utils/training.py::Trainer` trains all families via one loop; `collate_mode ∈
  {feta, preg, ensemble}` selects the forward signature. Pass a clinical-only
  `PatientGraphBuilder(include_ultrasound=False)` for the specialist config.
- Keep attention-dict keys stable across model edits — downstream code depends on them.
- The notebook is self-contained but reflects the **older single-etiology** pipeline;
  the `.py` modules are the source of truth for this session's work.
