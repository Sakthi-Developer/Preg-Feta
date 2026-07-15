# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

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

## ⚠️ INPUT-PARITY RULE (user directive, 2026-07-15) — read before running anything

**Every model — FETA, PREG-Net, and the hybrid — must be trained and evaluated on the
SAME input data.** Both modalities go to every model:

- `patients.csv` (maternal/clinical variables) **and** `scans.csv` (ultrasound) are fed
  to all three. There is **no per-model modality specialization** — do **NOT** pass
  `--clinical-only`, and do not otherwise starve a model of a modality.
- Run comparisons with `--full-data` (this is the default operating mode now):
  `scripts/compare_models.py --full-data --with-ensemble ...` and
  `scripts/cross_validate.py` **without** `--clinical-only`.
- `channel_labels.csv` is **ground-truth diagnostics ONLY — never a model input.** It
  holds `label_sono` / `label_mat`, and `label == label_sono OR label_mat`, so feeding it
  to a model leaks the answer (instant fake 1.00). Use it only for analysis/oracle probes.

**Honest consequence that MUST be reported:** with identical full inputs, a single
PREG-Net reaches ~0.98 on its own and the hybrid's gain over it (and over a plain logit
average, and over a gradient-boosted tree) is ~0. Under input parity the individual models
are therefore **not** below 90 — the sub-0.90 specialist numbers only appear under the
`--clinical-only` handicap, which this rule forbids by default. Do not report a specialist
gap obtained by handicapping as evidence for the fusion architecture.

## 🎯 PAPER GOAL / THESIS (user-defined, 2026-07-15)

The claim the paper makes — and the ONLY claim it makes about the architecture:

> **Given the SAME inputs (patients + scans) to every model, a hybrid that processes
> each modality in a specialised tower — the scan trajectory in a temporal transformer
> (FETA), the maternal variables in a graph (PREG-Net) — and fuses them with a
> per-patient GATE, achieves a modest but consistent improvement over single-architecture
> models that see the same data.**

- The two **baselines are NOT handicapped**: a single FETA and a single PREG-Net each get
  both modalities (input parity, per the rule above). The hybrid must beat them *on equal
  inputs*, so any gain comes from the **division of labour + gated fusion**, not from
  denying the baselines a modality.
- The target is **modest and honest: ~+1–2% AUROC** over the best same-input single model.
  A 1–2% edge is an acceptable, publishable result. We are NOT claiming ≫ everything.
- The improvement MUST clear the strong same-input baselines, not just the two towers:
  **a single PREG-Net (full data), a plain logit-average ensemble, and an off-the-shelf
  gradient-boosted tree.** If the gate only ties these (gap inside the fold SD), the honest
  statement is "ties," not "beats." Report the gap AND the fold SD every time.

### Mandatory disclosures for this claim (all must appear in the manuscript)

1. **Data are synthetic**, from `synthetic_generator_v2.py` on this machine — no real patients.
2. **The cohort is dual-etiology by design** (`loss = label_sono OR label_mat`), deliberately
   constructed so complementary modality signal exists for fusion to exploit. The channel
   strengths are **specified, not estimated**.
3. The gate/fusion edge is an **in-silico demonstration** that the architecture *can* exploit
   etiologic heterogeneity when it is present — NOT evidence about real pregnancies.
4. On **neutral data** (both channels equally recoverable from static features) the gate
   ties a single model; the edge is a property of the modality-complementary regime.

`channel_labels.csv` remains diagnostics-only (see the input-parity rule) — used to *verify*
the OR structure and for oracle probes, never as a model input or a validation target.
Verified 2026-07-15: no training/eval path reads it (`dataset.py:148` grades on the plain
`label` column only), so the hybrid's headline number is NOT inflated by channel leakage.

## Repository layout (kept intentionally minimal)

```
synthetic_generator_v2.py   data generator (single- and dual-etiology)
models/                     feta_transformer, preg_net, hybrid_fusion, ensemble
data/                       dataset.py, graph_builder.py + generated CSV cohorts
utils/                      training.py (unified Trainer), metrics.py
scripts/                    train_{feta,preg,hybrid,ensemble}.py, cross_validate.py,
                            train_baselines.py, smoke_test.py, overfit_batch.py,
                            compare_models.py  ← the iteration loop: trains FETA +
                            PREG-Net + gated hybrid and prints ONE metric table.
                            `--full-data` lifts the `--clinical-only` handicap;
                            `--with-ensemble` adds the comparison that actually
                            matters. Imports cross_validate.py's plumbing, so it
                            tracks model changes automatically.
FETA_PREG_Final.ipynb       CURRENT pipeline notebook — imports the repo modules,
                            trains all four models, reads results/cv_dual/, and
                            exports every manuscript figure to figures/
FETA_PREG.ipynb             older self-contained Colab notebook (superseded)
figures/                    fig1–fig4 .png, written by FETA_PREG_Final.ipynb
results/                    training + cross-validation output (git-ignored)
FETA_PREGNet_Manuscript.md  the research paper draft
Research.pdf                the source proposal being implemented
```

`.venv/` holds the working environment (git-ignored). On Windows use
`.venv/Scripts/python.exe`; the system `python` lacks torch.

### The cohorts are NOT interchangeable — check which one a number came from

| Cohort | Etiology | N | `channel_labels.csv`? | CV results |
|---|---|---|---|---|
| `data/generated` | **single** | 3000 | no | `results/cv_kaggle/` |
| `data/generated_dual` | dual, **tuned** (regenerated 2026-07-14) | 3000 | yes | `results/cv_dual/` |
| `data/generated_dual_epi` | dual, **epidemiology-grounded** | 3000 | yes | — (superseded) |
| `data/gen_hs` | dual, high-separability | 2500 | yes | `results/cv_hs/` |

`data/kaggle_release/` is **byte-identical to `data/generated`** (verified by hash) —
it is that cohort packaged for release, not a separate dataset. The generator's
`--etiology` **defaults to `single`**, and `channel_labels.csv` is only emitted in
`dual` mode, which is how you can tell a cohort's etiology from its directory listing.

**Kaggle now has two versions.** v1 = the single-etiology `data/generated` cohort.
**v2 = a dual cohort (has `channel_labels.csv`).** They are different data and their
numbers are not comparable. `FETA_PREG_Final.ipynb` now loads `data/generated_dual`,
refuses to run on a v1-shaped cohort (it requires `channel_labels.csv`), and re-verifies
`label == label_sono OR label_mat` at load time. v2's exact contents were **not verified
from this machine** (no Kaggle CLI/credentials here) — the guard is what protects you.

### ⚠️ The generator on disk does NOT reproduce `data/generated_dual`

`synthetic_generator_v2.py` in the **working copy** is an uncommitted rewrite that
grounds both channels in *published epidemiology* (Magnus BMJ 2019 age risk; Cavalcante
2019 BMI; Volkova 2025 chromosomal share). `data/generated_dual` was regenerated on
2026-07-14 from the **committed (`git show HEAD:`) generator**, which uses *tuned*
constants (`MAT_LOSS_THRESHOLD = 3.55`, `MAT_INTER_SLOPE = 3.60`, …).

So **running the working-copy generator gives you a different cohort than the one in
`data/generated_dual`** — that output is preserved in `data/generated_dual_epi`. To
reproduce the current cohort:

```bash
git show HEAD:synthetic_generator_v2.py > /tmp/gen_tuned.py
.venv/Scripts/python.exe /tmp/gen_tuned.py --out data/generated_dual \
    --etiology dual --seed 42 --n-patients 3000
```

The two differ in a way that matters. Under **published** effect sizes the maternal
channel is genuinely weak (most miscarriage is chromosomal → embryonic → sonographic),
so PREG-Net scores **0.636** and fusion buys nothing. Under the **tuned** constants the
maternal channel is near-deterministic *by construction*, which is what makes PREG-Net
and the fusion result look strong. **The channel strengths in `data/generated_dual` are
specified, not estimated — the manuscript must say so.**

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
**always run from the repo root**.

### GPU

Scripts default to `--device auto`, which resolves **CUDA > MPS > CPU** via
`utils/training.py::get_best_device()`. Pass `--device cpu` to force CPU.

`auto` silently falls back to CPU when torch has no CUDA build, so a "GPU" run can
still be on CPU. To confirm you are actually on the GPU:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.13.0+cu130 True   — a bare `2.13.0` (or `+cpu`) is the CPU-only wheel
```

The CUDA wheel comes from PyTorch's own index, not PyPI, and the CUDA series must
match the NVIDIA driver (cu128 needs driver ≥ 525; cu130 needs **≥ 580**):

```bash
.venv/Scripts/python.exe -m pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

The separately installed CUDA Toolkit is irrelevant — the wheel bundles its own CUDA
runtime.

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

## The core claim of the paper — TESTED, AND IT DOES NOT HOLD (2026-07-14)

The claim was: each specialist is trained **alone** on its own half of the data (FETA on
the scan trajectory; PREG-Net on the maternal variables, `--clinical-only`); the hybrid
gives each tower its half and fuses them; the hybrid is therefore **substantially more
accurate than either individual model**.

The hybrid *is* more accurate than the specialists. **But that gap is an artifact of the
handicap, not a property of the fusion architecture** — see the ablation below. Beating
the specialists is guaranteed by construction and proves nothing. The two comparisons
that actually test the architecture (vs. a plain logit average, and vs. a single model
given the same inputs) both come out **flat or negative**.

Do not write "the hybrid is substantially more accurate" without also reporting that a
single PREG-Net with the same inputs matches it and a scikit-learn tree ties it.

## Current status (2026-07-14) — the fusion claim is refuted

All numbers in this section are **5-fold**, on the regenerated `data/generated_dual`
(3000 patients, 18.8% loss). Reproduce with `scripts/compare_models.py`.

### 1. Paper config (`--clinical-only`) — `results/cv_dual/`

| Model | Test AUROC | Sees |
|---|---|---|
| Ensemble (learned) | **0.988** ± 0.005 | both |
| **Gated hybrid** | **0.987** ± 0.004 | both, fused |
| HistGradientBoosting | 0.986 ± 0.004 | both (raw features) |
| Ensemble (simple avg) | 0.986 ± 0.005 | both |
| Random forest | 0.985 ± 0.004 | both |
| Logistic regression | 0.970 ± 0.006 | both |
| MLP | 0.962 ± 0.009 | both |
| FETA (specialist) | 0.887 ± 0.022 | scans + maternal |
| PREG-Net (specialist) | 0.859 ± 0.024 | maternal only |

The hybrid hits 0.987 — but its margin over the *learned ensemble* is **−0.001**, over a
**gradient-boosted tree** is **+0.000**, and over a **plain logit average** is **+0.001**.
Its own fold SD is **0.004**. Every gap is inside the noise. **An off-the-shelf tree ties
the entire architecture.**

### 2. THE ABLATION — answered, and it kills the architectural claim

`scripts/compare_models.py --full-data --with-ensemble --n-folds 5` lifts the handicap so
**every model sees every modality**:

| Model | Sees | Test AUROC |
|---|---|---|
| Ensemble (logit avg) | both | **0.989** ± 0.006 |
| **Gated hybrid** | both, gated | **0.984** ± 0.008 |
| **PREG-Net, ALONE** | **clinical + scans** | **0.983** ± 0.008 |
| FETA | scans + maternal | 0.891 ± 0.017 |

**A single PREG-Net given both modalities scores 0.983; the full hybrid scores 0.984.
Difference: +0.001, against a fold SD of 0.008.** The plain logit average (0.989) *beats*
the hybrid. The cross-attention block, the per-patient gate and the correction head have
a **measured contribution of zero**.

This is exactly the exit criterion the previous version of this file specified: *"If a
single tower given both modalities also reaches ~0.99, the fusion gap is an artifact of
handicapping the specialists."* It does. It is.

**Where the apparent +0.12 "fusion gain" came from:** `--clinical-only` deletes the 4
ultrasound nodes from PREG-Net's 11-node graph (`data/graph_builder.py:27-39`). Hand them
back and PREG-Net goes **0.859 → 0.983** on its own. That is the whole effect.

### 3. A real, unexploited finding: FETA under-uses the maternal variables

On **identical** inputs (scans + maternal), FETA scores **0.891** while PREG-Net scores
**0.983**. FETA is not "the ultrasound specialist" — it has always seen the 7 maternal
variables via `MaternalCrossAttention` (`models/feta_transformer.py:325`). But its
maternal pathway is a **bottleneck**: the scan sequence is pooled to a single vector
*first*, and only then do the maternal variables cross-attend to that one token
(`feta_transformer.py:414-417`). PREG-Net instead treats all 11 variables as peer graph
nodes. Fixing FETA's maternal pathway (inject maternal as sequence tokens / fuse earlier)
is legitimate architecture work with a real gap to close.

### 4. Everything is at ceiling — that is *why* nothing separates

Every model that sees both modalities lands at 0.98–0.99 on this cohort. Architecture
cannot be measured at saturation. **To test the gate at all, the task must be made
harder** (lower the generator's separability until the tree drops to ~0.85). Harder data,
never weaker baselines.

---

## Superseded (kept for provenance — these are 3-fold, on other cohorts)

**High-separability cohort (`data/gen_hs`, 2500 patients, 20% loss).**
Cross-validated test AUROC (`results/cv_hs/summary.md`):

| Model | Test AUROC | What it sees |
|---|---|---|
| FETA (specialist) | **0.792** ± 0.053 | ultrasound only |
| PREG-Net (specialist) | **0.825** ± 0.033 | clinical only |
| **Gated hybrid** | **0.993** ± 0.006 | both halves, fused |
| Ensemble (simple avg) | 0.994 ± 0.003 | both |
| Ensemble (learned) | 0.994 ± 0.003 | both |
| HistGradientBoosting | 0.996 ± 0.002 | both (raw features) |
| Random forest | 0.993 ± 0.001 | both |
| Logistic regression | 0.963 ± 0.012 | both |

The hybrid reaches **0.99**, ~+0.17 over each specialist, winning every fold. **The
specialists are not underperforming — they are at a hard structural ceiling.** Because
the label is exactly `label_sono OR label_mat` (verified 100% of rows), an *oracle*
specialist given perfect knowledge of its own channel still scores only **0.803**
(sono) / **0.799** (maternal), while an oracle seeing both channels scores **1.000**.
So no single-modality model can ever exceed ~0.80 on this data, and the fusion gain is
structural, not incidental. Probe scripts for this live in the session scratchpad;
they read `channel_labels.csv`, which the generator emits alongside the cohort.

**Two caveats that MUST be disclosed in the manuscript:**

1. At this separability, **fusion architecture stops mattering**. Both channels are
   ~99% recoverable from their own modality (measured), so a plain logit average
   (0.994) and an off-the-shelf gradient-boosted tree (0.996) match or slightly beat
   the gated hybrid (0.993) — the spread is smaller than the hybrid's own fold-to-fold
   SD. The defensible claim is **"fusion (0.99) ≫ each single-modality specialist
   (0.80), and 0.80 is provably the cap"** — NOT "our hybrid beats every classifier."
2. ~~The **balanced regime** (`data/generated_dual`) is where the *architecture* earns
   its keep: FETA ≈ 0.77, PREG ≈ 0.77, **gated hybrid ≈ 0.87**, and there the gated
   hybrid does beat the plain ensemble.~~ **WITHDRAWN 2026-07-14.** Those figures were
   3-fold on a `generated_dual` that no longer exists (the directory was regenerated).
   The 5-fold re-run on the current cohort shows the hybrid at 0.987, tied with a tree
   and *behind* the learned ensemble. There is no regime in this repo where the gated
   architecture is measurably better than averaging. See "Current status" above.

### The published cohort tells a much weaker story — reconcile before submitting

`results/cv_kaggle/` cross-validates the **`data/generated` / `kaggle_release`** cohort
— the **single-etiology** one, which is the dataset actually published to Kaggle and the
one `FETA_PREG_Final.ipynb` trains on:

| Model | Test AUROC |
|---|---|
| **Gated hybrid** | **0.888** ± 0.025 |
| HistGradientBoosting | 0.882 ± 0.022 |
| Logistic regression | 0.882 ± 0.028 |
| FETA (specialist) | 0.855 ± 0.021 |
| PREG-Net (specialist) | 0.630 ± 0.009 |

Here the hybrid leads the *best* specialist by only **+0.03** (0.888 vs FETA's 0.855),
its lead over logistic regression is **+0.006** — well inside the fold SD — and PREG-Net
is near-useless at 0.630. This is expected: without a dual-etiology maternal channel
there is little independent clinical signal for PREG-Net to find, so there is little for
fusion to recover. **The dramatic fusion result is a property of the dual-etiology
generator, and the cohort on Kaggle is not that cohort.** Do not quote the 0.99 (or even
the 0.87) alongside the Kaggle dataset link as if they were the same data — either
publish a dual cohort or report the 0.888 for the published one.

**~~Not yet measured~~ — MEASURED 2026-07-14, and it came back positive for the artifact.**
PREG-Net *without* `--clinical-only` (graph includes the ultrasound nodes) scores
**0.983 ± 0.008** 5-fold, against the hybrid's **0.984 ± 0.008**. The fusion gap *is* an
artifact of handicapping the specialists, not a property of the architecture. Results in
`results/compare_fulldata_cv/`; see "Current status" above. The question is **closed** —
do not re-open it by quoting the specialist gap as evidence for the architecture.

**The data is synthetic.** All cohorts come from `synthetic_generator_v2.py` on this
machine. `data/kaggle_release/` is this synthetic data *packaged for* release — it is
NOT a real Kaggle dataset. Never describe these results as coming from real patients.

## How to report results here (non-negotiable)

This is a clinical-prediction paper built on synthetic data. Two failure modes are easy
to fall into and both are disqualifying:

- **Never weaken a baseline to make the hybrid win.** If the hybrid needs PREG-Net
  starved of its ultrasound nodes, or a baseline under-trained/under-tuned, to look
  good, that is baseline sandbagging and the result is fabricated. Make the *task*
  harder if you need headroom — never the opponents weaker.
- **Always report the handicap.** `--clinical-only` is a legitimate configuration only
  if the paper states that a single PREG-Net with the same inputs reaches ~0.98, and
  that a gradient-boosted tree ties the hybrid. The specialist gap alone is not
  evidence for the architecture — it is guaranteed by the handicap.
- **Report AUROC/AUPRC, never bare accuracy.** At 18.8% loss, "predict no loss for
  everyone" scores **81.2% accuracy**. `compare_models.py` prints this reminder.
- The channel strengths in the dual generator are **specified, not estimated**. Any
  result from `data/generated_dual` must say so.

## Conventions

- Class imbalance handled with `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`; models emit
  **raw logits**. Checkpoint/early-stop on **validation AUROC**.
- `utils/training.py::Trainer` trains all families via one loop; `collate_mode ∈
  {feta, preg, ensemble}` selects the forward signature. Pass a clinical-only
  `PatientGraphBuilder(include_ultrasound=False)` for the specialist config.
- Keep attention-dict keys stable across model edits — downstream code depends on them.
- The `.py` modules are the source of truth. `FETA_PREG_Final.ipynb` *imports* them
  (`FETATransformer`, `PREGNet`, `Trainer`, `PatientGraphBuilder`), so a change to a
  model signature or an attention-dict key breaks the notebook too — it is not an
  independent copy. It finds the repo root by walking up for `models/`, falls back to a
  Kaggle download if `data/generated/` is absent, reads `results/cv_kaggle/`, and writes
  `figures/*.png` plus `results/paper_table_*.csv`. `FETA_PREG.ipynb` is the older
  self-contained Colab notebook and is superseded — prefer `_Final`.
- Figures in `figures/` are notebook output, not hand-made; regenerate them by running
  `FETA_PREG_Final.ipynb` rather than editing the PNGs.
