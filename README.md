# FETA-PREGNet: First-Trimester Pregnancy-Loss Prediction

Two complementary deep models for first-trimester pregnancy-loss prediction and a
fusion of them, implementing the proposal in `Research.pdf`. All data is **synthetic**.

- **FETA-Transformer** — temporal attention over the longitudinal ultrasound sequence
  (the ultrasound specialist).
- **PREG-Net** — a knowledge-guided Graph Attention Network over clinical variables,
  pure PyTorch, run as a maternal-only clinical specialist (`--clinical-only`).
- **HybridFusionPredictor** — intermediate cross-attention fusion of the two, with a
  gated per-patient routing variant (`--gated`). This is the model of interest: it
  recovers losses that neither single-modality specialist sees alone.

## Layout

```
synthetic_generator_v2.py   synthetic data generator (single- / dual-etiology)
models/                     feta_transformer, preg_net, hybrid_fusion, ensemble
data/                       dataset.py, graph_builder.py + generated CSV cohorts
utils/                      training.py (unified Trainer), metrics.py
scripts/                    train_{feta,preg,hybrid,ensemble}, cross_validate,
                            train_baselines, smoke_test, overfit_batch
FETA_PREG.ipynb             self-contained notebook (older single-etiology pipeline)
FETA_PREGNet_Manuscript.md  research paper draft
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# (on macOS/Linux: source .venv/bin/activate && pip install -r requirements.txt)
```

The environment lives in `.venv/` (git-ignored). On Windows call
`.venv/Scripts/python.exe` directly — the system `python` does not have torch.

## Quick start (run from the repo root)

```bash
# 1. generate a dual-etiology cohort
.venv/Scripts/python.exe synthetic_generator_v2.py --out data/generated_dual \
    --etiology dual --seed 42 --n-patients 3000

# 2. verify the model wiring
.venv/Scripts/python.exe scripts/smoke_test.py

# 3. train the two specialists and the hybrid fusion
.venv/Scripts/python.exe scripts/train_feta.py   --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/feta   --epochs 60
.venv/Scripts/python.exe scripts/train_preg.py   --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/preg   --clinical-only --epochs 60
.venv/Scripts/python.exe scripts/train_hybrid.py --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --out-dir results/hybrid --clinical-only --gated --aux-weight 0.3 --epochs 80

# 4. cross-validate for stable numbers
.venv/Scripts/python.exe scripts/cross_validate.py --patients-csv data/generated_dual/patients.csv --scans-csv data/generated_dual/scans.csv --clinical-only --hybrid-gated --n-folds 5
```

## Dual-etiology synthetic design

`--etiology dual` generates each loss as the logical OR of an independent **sonographic**
channel (visible to FETA) and a **maternal** channel (visible to PREG-Net), so the two
modalities carry genuinely complementary information and their fusion is testable. This
is a deliberate modelling assumption for an in-silico benchmark, **not** a property of
real pregnancies; any reported fusion gains must be described as such. See `CLAUDE.md`
for the current tuning status and the full run reference.
