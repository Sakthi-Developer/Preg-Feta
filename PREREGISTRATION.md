# Pre-registration — does gated fusion beat same-input single models?

**Written 2026-07-16, BEFORE the harder cohort was generated and before any
fixed-FETA number was known.** The point of writing it first is that the
difficulty knobs and the success criteria cannot then be tuned toward each other.
If the criteria below are not met, the result is reported as stated here.

## Background: why the previous ~6 runs were uninformative

Two independent problems, both now identified:

1. **FETA's maternal pathway was disconnected** (found 2026-07-16, verified by
   zero gradient and bit-identical logits under 50x-different maternal input).
   `MaternalCrossAttention` used the pooled scan vector as its only key, and a
   softmax over one key is identically 1.0, which drops the maternal query out of
   the forward pass. Consequence: every "scans + maternal" row in every results
   table was really scans-only, and the hybrid saw maternal only through
   PREG-Net's graph — which is why hybrid ~= PREG-Net in every run.
2. **Ceiling saturation.** Every model that sees both modalities scores 0.98-0.99
   on the current cohorts. Architecture cannot be measured at saturation.

## Hypothesis

Given the SAME inputs (patients + scans) to every model, a hybrid that routes each
modality to a specialised tower (scan trajectory -> FETA; maternal variables ->
PREG-Net) and fuses them with a per-patient gate achieves a modest but consistent
AUROC improvement over single-architecture models that see the same data.

## Primary comparison

Gated hybrid vs. **the best same-input baseline**, defined as the maximum test
AUROC over:

- FETA-Transformer (fixed maternal pathway, full data)
- PREG-Net (full data, NOT `--clinical-only`)
- Plain logit-average ensemble
- HistGradientBoosting on raw features

Taking the **max** is deliberate: the hybrid must beat the strongest opponent, not
a convenient one. No baseline may be handicapped, under-trained, or under-tuned.

## Decision rule (fixed in advance)

Let `d = AUROC(hybrid) - AUROC(best same-input baseline)`, 5-fold CV, and let
`SD_pooled` be the pooled fold-to-fold SD of the two compared models.

| Outcome | Condition | Reported as |
|---|---|---|
| **SUPPORTED** | `d >= +0.010` AND `d >= 2 * SD_pooled` | "modest but consistent improvement" |
| **TIE** | `abs(d) < 2 * SD_pooled` | **"ties"** — explicitly NOT a win |
| **REFUTED** | `d <= -0.010` AND `abs(d) >= 2 * SD_pooled` | fusion underperforms |

A gap that is positive but smaller than `2 * SD_pooled` is a **TIE**, not a win.
This is the rule that the +0.001 and +0.005 gaps from previous runs fail.

Additionally, `d` must have the **same sign in at least 4 of 5 folds**. Prior runs
inverted sign across cohorts (`cv_dual_full`: ensemble 0.9889 > hybrid 0.9876;
`compare_fulldata`: hybrid 0.993 > ensemble 0.968) — the signature of noise.

## Cohort difficulty target (fixed in advance)

The harder cohort is tuned to **one** target, chosen before any hybrid number is
seen, and NOT adjusted afterward:

> **HistGradientBoosting test AUROC in [0.83, 0.87]** (target 0.85), 5-fold.

HistGradientBoosting is the tuning target *because it is a baseline, not our
model*. Difficulty is set by a reference classifier we are not advocating for, so
it cannot be steered toward the hybrid. Once the cohort hits this band, the knobs
are **frozen** and the comparison runs once.

The knobs may only reduce **recoverable signal**, never alter the OR structure:
- `--mat-noise-sd` — unobserved noise on the maternal risk score (caps how well
  ANY model recovers `label_mat` from maternal variables).
- `--sono-effect-scale` — shrinks abnormality magnitudes toward null (caps how
  well ANY model recovers `label_sono` from the scan trajectory).

Prevalence is held at ~20% and `MAT_LOSS_THRESHOLD` is recalibrated so the
maternal marginal rate stays at its tuned value. `label = label_sono OR label_mat`
is re-verified after generation.

## What will be reported regardless of outcome

1. The FETA maternal-pathway bug, its zero-gradient evidence, and the fact that it
   invalidates the previously reported "scans + maternal" FETA numbers.
2. Data are synthetic; channel strengths are **specified, not estimated**.
3. The gap AND the fold SD, every time.
4. If the result is a TIE, it is reported as a tie. The prior runs' honest summary
   ("no regime in this repo where the gated architecture is measurably better than
   averaging") stands unless this pre-registered test overturns it.

## Anti-goals

- Do not weaken a baseline to manufacture a gap.
- Do not re-tune difficulty after seeing the hybrid's number.
- Do not re-run with new seeds until a favourable draw appears. **One run.**
- `channel_labels.csv` stays diagnostics-only — never a model input.

---

# AMENDMENT 1 — equal-budget tuning sweep (written 2026-07-17)

**Status of the primary test above: RUN, VERDICT = TIE** (hybrid 0.825 vs
HistGradientBoosting 0.836; `d = -0.011`, 2*SD = 0.026, sign 3/5). That result is
**final and stands**. Nothing below can overturn it — this amendment defines a
*separate, exploratory* follow-up, and its outcome is reported as exploratory.

## Why this amendment exists

After seeing the TIE, the proposal was to optimize **only the hybrid** until it reached a
target. That is forbidden by Anti-goal 1 (it is baseline sandbagging: a gap produced by
tuning one arm measures tuning effort, not architecture). This amendment permits tuning
**only if every model is tuned under an identical budget**, fixed here before any sweep runs.

## Protocol (fixed in advance)

1. **Budget: 12 configurations per model.** Identical for hybrid, FETA, PREG-Net, and
   HistGradientBoosting. Random search over each model's space, seed 0. The ensemble is
   built from the tuned FETA + tuned PREG-Net (it has no parameters of its own).
   **The hybrid gets no extra configs, no extra epochs, and no hand-picked runs.**
2. **Selection signal: mean best VALIDATION AUROC over folds 1-3** (`trainer.best_auroc`,
   the inner `val` split from `make_folds`). **Test AUROC is not computed, looked at, or
   used for selection during the sweep.** Tuning on test would make the 0.836 reference
   meaningless.
3. **Order is irrelevant to fairness but is fixed anyway:** hybrid first (per user
   request), then FETA, PREG-Net, HGB. **Stopping after the hybrid and declaring a result
   is not permitted** — a partial sweep is reported as "no result", not as a win.
4. **Final scoring: ONE 5-fold test run** with each model at its selected config, scored
   by `scripts/analyze_prereg.py` under the SAME decision rule as the primary test
   (`d >= +0.010` AND `d >= 2*SD_pooled` AND sign in >= 4/5 folds).
5. **Cohort and difficulty knobs stay FROZEN.** `data/gen_hard`, unchanged. No regeneration.
6. **Confirmation required.** Any SUPPORTED outcome must replicate on a fresh seed
   (`--seed 43`) before it is written into the manuscript. A win on one seed is a draw.

## Honest expectation, recorded before the sweep

HistGradientBoosting with full raw features scores **0.836**, which is the practical
estimate of recoverable signal in `data/gen_hard` after `--mat-noise-sd 3.0` and
`--sono-effect-scale 0.35`. **A hybrid at ~0.95 is therefore not an expected outcome of
tuning; it is a leakage alarm** — most likely `channel_labels.csv`, which stays
diagnostics-only. Realistic tuning movement is ~±0.01-0.02 for every arm, which is inside
`2*SD_pooled`. **The most probable outcome of this amendment is that the TIE survives**,
because tuning tends to lift all arms together.

## Reported regardless of outcome

- Every config tried, per model, with its val AUROC — not just the winner.
- That the sweep is **post-hoc / exploratory**, run after the primary test was a TIE.
- If the hybrid wins only because the sweep found it more configs, or if any baseline was
  given a smaller space, the result is void.
