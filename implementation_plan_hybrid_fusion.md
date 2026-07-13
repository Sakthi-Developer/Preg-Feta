# Implementation Plan: Hybrid Cross-Attention Fusion (FETA ⇄ PREG-Net)

Last updated: 2026-07-10

## 1. Decision and rationale

Replace/augment the current **late-fusion** ensemble (`EnsemblePredictor`, which only
averages or linearly combines the two final logits) with an **intermediate-fusion
hybrid**: FETA-Transformer and PREG-Net exchange information *mid-network* via
bidirectional cross-attention, then produce a single combined prediction.

**Why intermediate fusion (the "talking" model):**

- Literature review of multimodal graph+sequence models in healthcare finds
  intermediate/attention fusion is the dominant approach (~81% of models) and the
  top performer (mean AUC ~0.95, up to 1.00), beating both early fusion and late
  fusion. Cross-modal conditional attention beat simple late fusion by ~+7% accuracy
  in a head-to-head study. (Frontiers in AI 2025 review; MDPI comparative study.)
- It matches the project's research story: a single unified architecture where the
  temporal model and the graph model attend to each other, with a novel cross-modal
  attention artifact for the manuscript.
- Late fusion (`EnsemblePredictor`) is retained as a simpler baseline/fallback row.

**Why we add an anti-collapse safeguard (not in the reviews, but required here):**
Measured on the current test split, the PREG branch *inside* the trained averaged
ensemble collapsed to AUROC 0.907 (vs 0.986 standalone) — FETA dominated and PREG
became dead weight. Any joint architecture will repeat this unless we explicitly
keep both branches predictive. We add an auxiliary per-branch loss.

## 2. Goal and success criteria

Produce one output that is **provably** better than either model alone AND better than
the existing late-fusion ensemble.

Success = ALL of:

1. On a harder/larger cohort (see Phase 5), hybrid mean test AUROC exceeds both
   standalone models and the late-fusion ensemble, with a **paired bootstrap 95% CI
   for the difference that excludes 0** (via `scripts/statistical_analysis.py`).
2. **No branch collapse**: standalone-equivalent AUROC of each branch (from the
   auxiliary heads) stays within ~0.02 of the separately trained models.
3. All existing contracts preserved (Section 3); `smoke_test.py` and
   `overfit_batch.py` pass.

Note: on the *current* saturated cohort (both models ~0.99, only ~8/122 test patients
"in play", error correlation 0.40) no fusion topology can demonstrate a real gain.
The harder-cohort work in Phase 5 is a hard prerequisite for criterion (1), not optional.

## 3. Contracts that MUST stay stable

- **Pure PyTorch** — no `torch_geometric`. Cross-attention is hand-written `nn.Linear`
  + `scaled_dot_product`-style math, consistent with the existing hand-rolled GAT.
- **Standalone sub-model `forward()` signatures and attention-dict keys are frozen.**
  `scripts/export_explainability.py`, `scripts/smoke_test.py`, `scripts/train_feta.py`,
  and `scripts/train_preg.py` call `FETATransformer`/`PREGNet` directly and depend on:
  - FETA → `(logits, {'temporal_pooling_weights', 'cross_attention_weights'})`
  - PREG → `(logits, {'node_importance', 'edge_attention'})`
  The refactor in Phase 1 is behavior-preserving; these are unchanged.
- **Ensemble-style output dict schema** consumed by `scripts/train_ensemble.py`
  (`logits`, `feta_logits`, `preg_logits`, `feta_attention`, `preg_attention`). The
  hybrid returns the same keys plus a new `cross_modal_attention` key.
- **`collate_mode="ensemble"` path** in `utils/training.py::Trainer` is reused
  unchanged for the forward pass (it already passes both temporal and graph tensors).
  The only Trainer addition is an opt-in auxiliary-loss weight (default 0 = no change).

## 4. Architecture

Fusion point = each model's pre-pooling token/node sequence, where both are `(B, *, d)`
with `d = 64`:

- FETA: token sequence `h_feta (B, T=5, d)` after `transformer_encoder`, before `attn_pool`.
- PREG: node embeddings `h_preg (B, N=27, d)` after the GAT stack, before `readout`.

```
temporal + maternal ─► FETA.encode_tokens ─► h_feta (B,T,d) ─┐
                                                             ├─► CrossModalFusion ─► h_feta', h_preg'
graph tensors ───────► PREG.encode_nodes  ─► h_preg (B,N,d) ─┘        │
                                                                      ├─ h_feta' ─► FETA.head ─► z_feta ─► feta_logits (aux)
                                                                      └─ h_preg' ─► PREG.head ─► z_preg ─► preg_logits (aux)
                                                     z_feta ⊕ z_preg ⊕ [feta_logits, preg_logits] ─► fusion head ─► final logit
```

**CrossModalFusion (new module):** two multi-head cross-attention passes, each with
residual + LayerNorm.
- FETA→PREG: query = `h_feta` tokens, key/value = `h_preg` nodes, `key_padding_mask` =
  `node_mask` (ignore padded nodes). Output updates `h_feta`.
- PREG→FETA: query = `h_preg` nodes, key/value = `h_feta` tokens, `key_padding_mask` =
  `temporal_mask` (ignore padded scans). Output updates `h_preg`.
- Cost is trivial (T=5 × N=27 attention matrix); adds ~30–50k params total.
- Masking guard: every patient has ≥2 scans and ≥ maternal nodes, so no query ever has
  all keys masked; still assert to avoid silent NaNs.

**Fusion head:** `LayerNorm(2d)` over `[z_feta ⊕ z_preg]`, concat the two branch logits,
then `Linear(2d+2, d) → GELU → Dropout → Linear(d, 1)`. Including the branch logits
means the head strictly generalizes the current learned-logit fusion.

**Anti-collapse:** total loss = `BCE(final) + λ·(BCE(feta_logits) + BCE(preg_logits))`,
λ default 0.3. Keeps both branches individually predictive and stabilizes training.

## 5. Component-by-component changes

### 5.1 `models/feta_transformer.py` — behavior-preserving split
- [ ] Add `encode_tokens(temporal_features, gestational_ages, temporal_mask)` → returns
  `h_feta (B,T,d)` (steps: modality proj, positional encoding, input norm/dropout,
  transformer encoder).
- [ ] Add `head(h_feta, temporal_mask, maternal_features)` → runs `attn_pool` +
  `maternal_cross_attn` + `classifier`; returns `(logits, {temporal_pooling_weights,
  cross_attention_weights}, z)` where `z` is the pooled embedding.
- [ ] `forward()` = `head(encode_tokens(...))` — identical output to today.

### 5.2 `models/preg_net.py` — behavior-preserving split
- [ ] Add `encode_nodes(node_features, edge_index, node_types, node_mask)` → returns
  `h_preg (B,N,d)` and the `edge_attentions` list (the GAT stack).
- [ ] Add `head(h_preg, node_mask)` → runs `readout` + `classifier`; returns
  `(logits, {node_importance}, z)`.
- [ ] `forward()` = compose the two, re-attaching `edge_attention` — identical output.

### 5.3 `models/hybrid_fusion.py` — new file
- [ ] `CrossModalFusion(d_model, n_heads, dropout)` module (bidirectional, masked).
- [ ] `HybridFusionPredictor(feta_kwargs, preg_kwargs, n_heads, dropout,
  bidirectional=True)` holding a `FETATransformer`, `PREGNet`, `CrossModalFusion`,
  and the fusion head.
- [ ] `forward(...)` with the **same signature as `EnsemblePredictor.forward`**; returns
  dict `{logits, feta_logits, preg_logits, feta_attention, preg_attention,
  cross_modal_attention, fusion_weights=None}`.
- [ ] `cross_modal_attention = {'feta_to_preg': (B,H,T,N), 'preg_to_feta': (B,H,N,T)}`.

### 5.4 `models/__init__.py`
- [ ] Export `HybridFusionPredictor` and `CrossModalFusion`.

### 5.5 `utils/training.py::Trainer` — opt-in auxiliary loss
- [ ] Add `aux_loss_weight: float = 0.0` constructor arg.
- [ ] In `train_epoch`/`validate_epoch`: if output is a dict containing `feta_logits`
  and `preg_logits` and `aux_loss_weight > 0`, add
  `aux_loss_weight * (BCE(feta_logits) + BCE(preg_logits))` to the main loss.
- [ ] Fully backward-compatible: default 0.0 leaves all existing training runs unchanged.

### 5.6 `scripts/train_hybrid.py` — new entry point
- [ ] Mirror `scripts/train_ensemble.py`; build `HybridFusionPredictor`,
  `collate_mode="ensemble"`, pass `aux_loss_weight` (CLI `--aux-weight`, default 0.3).
- [ ] Reuse the ensemble `evaluate_loader` (needs `feta_logits`/`preg_logits`, present).
- [ ] Output dir `results/hybrid`; save `--bidirectional/--unidirectional` in run config.

### 5.7 `scripts/cross_validate.py` — register the model
- [ ] Add `"hybrid"` to `DEEP_MODELS`.
- [ ] In `build_deep_model`: `if model_name == "hybrid": return HybridFusionPredictor(), "ensemble"`.
- [ ] Note: CV path uses the Trainer without aux loss unless we thread `aux_loss_weight`
  through `run_deep_fold`; add that thread so CV trains the hybrid the same way as
  `train_hybrid.py`.

### 5.8 Test/verification hooks
- [ ] `scripts/smoke_test.py`: add a `HybridFusionPredictor` forward assertion
  (`logits (2,1)`, presence of `cross_modal_attention` with correct shapes).
- [ ] `scripts/overfit_batch.py`: add `"hybrid"` to `--mode` choices and `build_model`,
  include in the `all` sweep.

## 6. Phased execution

### Phase 1 — Refactor (no behavior change)
- [ ] 5.1 + 5.2 splits. Run `python scripts/smoke_test.py` and
  `python scripts/overfit_batch.py --mode all --cpu` → must still pass unchanged.
  Confirms standalone contracts intact before adding anything.

### Phase 2 — Build the hybrid
- [ ] 5.3 + 5.4 (`CrossModalFusion`, `HybridFusionPredictor`, exports).
- [ ] 5.8 smoke assertion. Run smoke test.

### Phase 3 — Train integration
- [ ] 5.5 Trainer aux-loss hook.
- [ ] 5.6 `train_hybrid.py`.
- [ ] 5.8 overfit hook. Run `python scripts/overfit_batch.py --mode hybrid --cpu`
  (must reduce one-batch loss → catches wiring bugs).
- [ ] Short smoke train: `python scripts/train_hybrid.py --epochs 3 --out-dir results/hybrid_smoke --device cpu`.

### Phase 4 — Full run on current data (sanity, not the proof)
- [ ] `python scripts/train_hybrid.py --epochs 80 --device cpu` → `results/hybrid`.
- [ ] Confirm branches did NOT collapse (aux-head AUROCs ≈ standalone). Expect fused
  AUROC ≈ 0.99 (saturated); this validates the machinery, not the hypothesis.
- [ ] 5.7 register in CV; run CV smoke.

### Phase 5 — Harder cohort + statistical proof (the actual experiment)
- [ ] Inspect `synthetic_generator_v2.py` difficulty knobs (noise, class overlap,
  `--n-patients`); produce a harder cohort where standalone models land ~0.83–0.88
  with *complementary* errors, plus a larger test set (more positives).
- [ ] Retrain PREG, FETA, late-fusion ensemble, and hybrid on the harder cohort.
- [ ] `python scripts/cross_validate.py --models preg,feta,ensemble,learned_ensemble,hybrid ...`.
- [ ] `python scripts/statistical_analysis.py` → paired bootstrap AUROC-difference CIs
  for hybrid vs {FETA, PREG, ensemble}. Success = CIs exclude 0 (criterion 1).

### Phase 6 — Explainability + paper artifacts
- [ ] New exporter for `cross_modal_attention` (which scans attended to which graph
  variables, and vice versa) — additive; does not touch existing exporters.
- [ ] Add hybrid rows to `compare_models.py` and `generate_paper_artifacts.py`.
- [ ] Ablation rows: average vs learned-logit vs hybrid(no aux) vs hybrid(+aux) vs
  hybrid(unidirectional) — feeds the pending ablation section in the main plan.

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| One branch collapses (measured, real) | Auxiliary per-branch BCE loss (λ=0.3); assert aux-head AUROC ≈ standalone in Phase 4. |
| Overfitting from added params | Small fusion head, dropout, AdamW weight decay; only justified once cohort is larger (Phase 5). |
| Cross-attention NaNs on all-masked keys | Guaranteed ≥1 valid key (≥2 scans, maternal nodes); assert masks have a valid entry. |
| "Improvement" is noise | Paired bootstrap CIs on a larger test set (Phase 5), never point estimates. |
| Breaking standalone explainability | Phase 1 is behavior-preserving; smoke/overfit gate every phase. |

## 8. Out of scope / deferred
- Fully merged single-tower model (destroys the two named, explainable models).
- Sequential/cascade fusion (weaker; copycat-collapse prone) — kept only as an optional
  ablation if time permits.
- Real clinical data (project is synthetic-only by design).
