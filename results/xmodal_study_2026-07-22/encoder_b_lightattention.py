"""ENCODER B -- does a LightAttention sequence encoder add anything over the tree views?

Completes the guide's Phase 2 exactly as written:

    Encoder A   tree on the base feature table
    Encoder B   LightAttention on the scan sequence      <- the missing piece
    Encoder C   tree on the cross-modal features
    Gated fusion with learned per-patient weights

LightAttention = FETATransformer(d_model=24, n_layers=1) = 12,441 parameters, which is
what the guide's "~10,000 parameters" decodes to. Measured on this repo's own class:
7,249 at d_model=16, 12,441 at d_model=24, 18,721 at d_model=32; the deployed FETA is
88,193 (d_model=64, n_layers=2). So this is FETA shrunk ~7x, not a new architecture.

WHAT THIS ISOLATES
------------------
Encoder B's logits are trained ONCE per outer fold and then shared by both the
XGBoost-expert and CatBoost-expert versions of the hybrid. So the 2-expert vs 3-expert
delta is attributable to the sequence encoder alone, with the tree component held fixed
-- the same "read it down a column" discipline that the learner/context confound
requires (comparing across expert types would reintroduce exactly that confound).

Rule 1  Encoder B is built with inject_maternal_tokens=True, so it sees BOTH modalities
        like every other arm. Parity is a property of the wiring; check_input_parity.py
        is run separately after this.
Rule 2  the base tree table is build_feature_table_fast via the benchmark's load().
Rule 5  the same 5x2 folds as every other number in this study.
Rule 7  the comparator keeps the same component and the same features; the hybrid gets
        strictly MORE compute, never less.

Normalisation statistics are fit on the training split of each fold only.

DECISION RULE -- unchanged, applied verbatim:
    REAL only if |mean delta| > 2 x pooled fold SD AND sign agrees on >= 8/10 folds.
    Otherwise TIE, whatever the sign.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "regime_study_2026-07-19"))

from catboost import CatBoostClassifier  # noqa: E402
from models.feta_transformer import FETATransformer  # noqa: E402
from published_families_benchmark import COH, load  # noqa: E402
from xmodal_features import (SEED, _logit, build_xmodal_block, verdict,  # noqa: E402
                             xgb)

US = ["FHR", "CRL", "GS", "YSD"]
D_MODEL, N_LAYERS, N_HEADS = 24, 1, 2


def cat() -> CatBoostClassifier:
    return CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False)


# ---------------------------------------------------------------- sequences
def build_sequences(pat: pd.DataFrame, scans: pd.DataFrame):
    """Padded (N,T,4) measurements, (N,T) GA, (N,T) mask, aligned to patients.csv order.

    Raw values -- per-fold normalisation happens in the loop, on train statistics only.
    """
    sc = scans.sort_values(["patient_id", "gestational_age_weeks"])
    order = {p: i for i, p in enumerate(pat["patient_id"])}
    T = int(sc.groupby("patient_id").size().max())
    N = len(pat)

    X = np.zeros((N, T, len(US)), dtype=np.float32)
    G = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    for pid, grp in sc.groupby("patient_id", sort=False):
        i = order[pid]
        k = min(len(grp), T)
        X[i, :k] = grp[US].values[:k].astype(np.float32)
        G[i, :k] = grp["gestational_age_weeks"].values[:k].astype(np.float32)
        M[i, :k] = 1.0
    return np.nan_to_num(X), G, M, T


def fit_light_encoder(X, G, Msk, Mat, y, tr, epochs: int, seed: int = 0):
    """Train LightAttention on one split. Fixed epochs, no early stop -- same protocol
    as published_families_benchmark, so no arm gets a selection advantage."""
    torch.manual_seed(seed)
    net = FETATransformer(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                          n_us_features=len(US), n_maternal=Mat.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    xt, gt, mt, at, yt = (torch.tensor(a, dtype=torch.float32)
                          for a in (X, G, Msk, Mat, y))
    idx = torch.tensor(tr)
    net.train()
    for _ in range(epochs):
        for b in idx[torch.randperm(len(idx))].split(64):
            opt.zero_grad()
            out, _ = net(xt[b], gt[b], mt[b], at[b])
            lossf(out.squeeze(-1), yt[b]).backward()
            opt.step()
    net.eval()
    return net, (xt, gt, mt, at)


def light_logits(net, tensors, rows) -> np.ndarray:
    xt, gt, mt, at = tensors
    outs = []
    with torch.no_grad():
        for b in torch.tensor(rows).split(512):
            o, _ = net(xt[b], gt[b], mt[b], at[b])
            outs.append(o.squeeze(-1).numpy())
    return np.concatenate(outs)


def fit_gate(expert_logits_tr, gate_X_tr, y_tr, seed: int = 0):
    """Softmax gate over K experts, trained on out-of-fold expert logits.

    Expert logits are standardised per expert on TRAIN statistics first. Without this
    the gate weight is confounded with logit MAGNITUDE: separately-trained experts do
    not share a scale, so a wide-logit expert dominates the weighted sum even at equal
    weight, and the learned weights stop meaning "how much do I trust this expert".
    (Measured: unstandardised, the gate put 0.697 of its mass on the weakest expert.)
    The published GATED arm does not need this because its towers are trained jointly
    and therefore already share a scale. Ranking metrics are unaffected by the affine
    transform itself; only the gate's ability to express a preference changes.
    """
    K = len(expert_logits_tr)
    sc = StandardScaler().fit(gate_X_tr)
    lsc = StandardScaler().fit(np.column_stack(expert_logits_tr))
    Gt = torch.tensor(sc.transform(gate_X_tr), dtype=torch.float32)
    L = torch.tensor(lsc.transform(np.column_stack(expert_logits_tr)),
                     dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)

    torch.manual_seed(seed)
    gate = nn.Sequential(nn.Linear(Gt.shape[1], 16), nn.ReLU(), nn.Linear(16, K))
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    idx = torch.arange(len(y_tr))
    for _ in range(200):
        for b in idx[torch.randperm(len(idx))].split(256):
            opt.zero_grad()
            w = torch.softmax(gate(Gt[b]), -1)
            lossf((w * L[b]).sum(-1), yt[b]).backward()
            opt.step()
    return gate, sc, lsc


def apply_gate(gate, sc, lsc, expert_logits_te, gate_X_te):
    with torch.no_grad():
        w = torch.softmax(
            gate(torch.tensor(sc.transform(gate_X_te), dtype=torch.float32)), -1
        ).numpy()
    return (w * lsc.transform(np.column_stack(expert_logits_te))).sum(-1), w


# ---------------------------------------------------------------- run
def main() -> None:
    ap = argparse.ArgumentParser()
    # 150 matches published_families_benchmark's neural budget, so Encoder B is not
    # handicapped relative to the fusion arms it is being compared against (rule 7).
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    pat = pd.read_csv(REPO / COH / "patients.csv")
    raw = pd.read_csv(REPO / COH / "scans.csv")
    Mat, Xs, y, Xbase = load(COH)
    Xmod = np.nan_to_num(build_xmodal_block(pat, raw).values,
                         nan=0.0, posinf=0.0, neginf=0.0)
    Xseq, GA, Msk, T = build_sequences(pat, raw)

    probe = FETATransformer(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                            n_us_features=len(US), n_maternal=Mat.shape[1])
    n_par = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"cohort {COH}   n={len(y)}   max visits T={T}")
    print(f"Encoder B  LightAttention d_model={D_MODEL} layers={N_LAYERS}  "
          f"{n_par:,} params  (guide's budget ~10,000; deployed FETA 88,193)")
    print(f"epochs={args.epochs}   folds={args.folds}/10\n")

    splits = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(Xs, y)
    )[: args.folds]

    arms = ["A+C/xgb", "A+B+C/xgb", "A+C/cat", "A+B+C/cat", "B-alone"]
    auroc = {a: [] for a in arms}
    auprc = {a: [] for a in arms}
    gate_b = []

    for k, (tr, te) in enumerate(splits, 1):
        t0 = time.time()
        # ---- per-fold normalisation, train statistics only ----
        valid = Msk[tr].astype(bool)
        mu = np.array([Xseq[tr][..., j][valid].mean() for j in range(len(US))])
        sd = np.array([Xseq[tr][..., j][valid].std() or 1.0 for j in range(len(US))])
        Xn = ((Xseq - mu) / sd) * Msk[..., None]
        ms = StandardScaler().fit(Mat[tr])
        Mn = ms.transform(Mat)

        # ---- Encoder B: inner-OOF logits for the gate, then a full-train refit ----
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        oof = {e: np.zeros(len(tr)) for e in ("A", "B", "C", "Acat", "Ccat")}
        for itr, ite in inner.split(Xbase[tr], y[tr]):
            gtr, gte = tr[itr], tr[ite]
            net, tens = fit_light_encoder(Xn, GA, Msk, Mn, y, gtr, args.epochs, seed=0)
            oof["B"][ite] = light_logits(net, tens, gte)
            oof["A"][ite] = _logit(xgb().fit(Xbase[gtr], y[gtr])
                                   .predict_proba(Xbase[gte])[:, 1])
            oof["C"][ite] = _logit(xgb().fit(Xmod[gtr], y[gtr])
                                   .predict_proba(Xmod[gte])[:, 1])
            oof["Acat"][ite] = _logit(cat().fit(Xbase[gtr], y[gtr])
                                      .predict_proba(Xbase[gte])[:, 1])
            oof["Ccat"][ite] = _logit(cat().fit(Xmod[gtr], y[gtr])
                                      .predict_proba(Xmod[gte])[:, 1])

        net, tens = fit_light_encoder(Xn, GA, Msk, Mn, y, tr, args.epochs, seed=0)
        te_l = {
            "B": light_logits(net, tens, te),
            "A": _logit(xgb().fit(Xbase[tr], y[tr]).predict_proba(Xbase[te])[:, 1]),
            "C": _logit(xgb().fit(Xmod[tr], y[tr]).predict_proba(Xmod[te])[:, 1]),
            "Acat": _logit(cat().fit(Xbase[tr], y[tr]).predict_proba(Xbase[te])[:, 1]),
            "Ccat": _logit(cat().fit(Xmod[tr], y[tr]).predict_proba(Xmod[te])[:, 1]),
        }

        combos = {
            "A+C/xgb": ["A", "C"],
            "A+B+C/xgb": ["A", "B", "C"],
            "A+C/cat": ["Acat", "Ccat"],
            "A+B+C/cat": ["Acat", "B", "Ccat"],
        }
        for arm, keys in combos.items():
            gate, gsc, lsc = fit_gate([oof[e] for e in keys], Xbase[tr], y[tr], seed=0)
            p, w = apply_gate(gate, gsc, lsc, [te_l[e] for e in keys], Xbase[te])
            auroc[arm].append(roc_auc_score(y[te], p))
            auprc[arm].append(average_precision_score(y[te], p))
            if arm == "A+B+C/cat":
                gate_b.append(float(w[:, 1].mean()))

        auroc["B-alone"].append(roc_auc_score(y[te], te_l["B"]))
        auprc["B-alone"].append(average_precision_score(y[te], te_l["B"]))
        print(f"  fold {k:2d}/{len(splits)}  A+B+C/cat={auroc['A+B+C/cat'][-1]:.3f}  "
              f"B-alone={auroc['B-alone'][-1]:.3f}  "
              f"gate_B={gate_b[-1]:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 66)
    print(f"{'arm':<14}{'AUROC':>16}{'AUPRC':>9}")
    print("-" * 66)
    for a in arms:
        print(f"{a:<14}{np.mean(auroc[a]):>8.3f} +/- {np.std(auroc[a]):.3f}"
              f"{np.mean(auprc[a]):>9.3f}")
    print("=" * 66)

    print("\nDOES ENCODER B ADD ANYTHING? (tree component held fixed)")
    print(f"{'comparison':<34}{'delta':>8}{'2*SD':>8}{'sign':>7}  verdict")
    print("-" * 66)
    rows = []
    for tag in ("xgb", "cat"):
        v = verdict(auroc[f"A+B+C/{tag}"], auroc[f"A+C/{tag}"])
        rows.append(dict(comparison=f"A+B+C/{tag} - A+C/{tag}", **v))
        print(f"{'A+B+C vs A+C  (' + tag + ')':<34}{v['delta']:>+8.3f}"
              f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    if args.folds == 10:
        prev = pd.read_csv(HERE / "xmodal_per_fold.csv").pivot(
            index="fold", columns="arm", values="auroc")
        v = verdict(auroc["A+B+C/cat"], prev["CatBoost/+xmod"].values)
        rows.append(dict(comparison="A+B+C/cat - CatBoost/+xmod", **v))
        print(f"{'A+B+C/cat vs one joint CatBoost':<34}{v['delta']:>+8.3f}"
              f"{2*v['pooled_sd']:>8.3f}{v['sign']:>7}  {v['verdict']}")

    print(f"\nmean gate weight on Encoder B: {np.mean(gate_b):.3f} of 3 experts "
          f"(equal share = 0.333)")

    pd.DataFrame([dict(arm=a, fold=i + 1, auroc=auroc[a][i], auprc=auprc[a][i])
                  for a in arms for i in range(len(splits))]).to_csv(
        HERE / "encoder_b_per_fold.csv", index=False)
    pd.DataFrame(rows).to_csv(HERE / "encoder_b_verdicts.csv", index=False)
    print("wrote encoder_b_per_fold.csv and encoder_b_verdicts.csv")


if __name__ == "__main__":
    main()
