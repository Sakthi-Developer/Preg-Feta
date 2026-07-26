"""BENCHMARK: published multimodal-fusion approach FAMILIES on one common cohort.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
It is NOT a comparison against published AUROC numbers. Those were obtained on different
cohorts (real patients, different prevalence, different difficulty), and AUROC is not
comparable across datasets -- a higher number on an easier cohort says nothing. Quoting
"they got 0.969, we got 0.836" would be meaningless and is exactly the comparison a
reviewer rejects.

What IS valid, and what this does: re-implement the ARCHITECTURAL FAMILY behind each
published design and run every one on the SAME cohort, the SAME folds, the SAME features,
and the SAME training budget. Then the differences are attributable to architecture.

Families covered (representative of the current literature):

  LATE-AVG     late fusion by averaging tower outputs      (classic ensemble fusion)
  LATE-STACK   late fusion with learned weights            (stacked generalisation)
  EARLY        early fusion: concatenate raw features -> MLP
  INTERMEDIATE joint MLP over both towers' representations (intermediate/joint fusion)
  GATED        per-patient gating over towers              (attention/gated fusion)
  MoE          mixture-of-experts: softmax gate over experts, trained jointly
  DEEP2GBM     deep towers as feature extractors -> gradient boosting on the embeddings
               (the design used by the best-performing published EPL model, where a deep
                net measures biometrics and a boosted tree makes the decision)
  GBM          gradient boosting on raw features           (no fusion at all)
  LOGREG       logistic regression on raw features         (floor)

Every neural arm uses identical tower architecture, width, dropout, optimiser, epochs and
seeds, so no arm is advantaged by tuning effort. Trees use the repository's own feature
table (build_feature_table_fast), NOT ad-hoc features -- using ad-hoc features previously
understated the tree by ~0.019 and manufactured a false win for a neural arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MEAS = ["FHR", "CRL", "GS", "YSD"]
COH = "data/gen_hard"
EPOCHS, SEEDS, HID = 150, 3, 32


# ---------------------------------------------------------------- data
def load(coh: str):
    pat = pd.read_csv(REPO / coh / "patients.csv")
    raw = pd.read_csv(REPO / coh / "scans.csv")
    sc = raw.sort_values(["patient_id", "gestational_age_weeks"])
    ids, y = pat["patient_id"], pat["label"].values.astype(np.float32)

    mat = pat.drop(columns=["patient_id", "label"])
    cat = [c for c in mat.columns if not pd.api.types.is_numeric_dtype(mat[c])]
    mat = pd.get_dummies(mat, columns=cat, dummy_na=True) if cat else mat

    sc = sc.copy()
    sc["visit"] = sc.groupby("patient_id").cumcount()
    flat = sc.pivot_table(index="patient_id", columns="visit",
                          values=MEAS + ["gestational_age_weeks"])
    flat.columns = [f"{a}_v{b}" for a, b in flat.columns]
    g = sc.groupby("patient_id")
    for m in MEAS:
        flat[f"{m}_mean"], flat[f"{m}_last"] = g[m].mean(), g[m].last()
    flat["n_visits"] = g.size()
    scan = flat.reindex(ids).reset_index(drop=True)

    from scripts.scaling_sweep import build_feature_table_fast
    ft = build_feature_table_fast(pat, raw)
    assert np.array_equal(ft["label"].values.astype(np.float32), y), "row order mismatch"
    Xtree = ft.drop(columns=["patient_id", "label"]).astype(float).values

    return (np.nan_to_num(mat.astype(float).values),
            np.nan_to_num(scan.astype(float).values), y, Xtree)


# ---------------------------------------------------------------- nets
class Tower(nn.Module):
    def __init__(self, d, h=HID):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        r = self.f(x)
        return r, self.head(r).squeeze(-1)


class Net(nn.Module):
    """One class, many fusion families -- so towers are provably identical across arms."""

    def __init__(self, ds, dm, family):
        super().__init__()
        self.family = family
        if family == "EARLY":
            self.mlp = nn.Sequential(nn.Linear(ds + dm, HID), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
            return
        self.s, self.m = Tower(ds), Tower(dm)
        if family == "INTERMEDIATE":
            self.head = nn.Sequential(nn.Linear(2 * HID, HID), nn.ReLU(), nn.Linear(HID, 1))
        elif family in ("GATED", "MoE"):
            self.gate = nn.Sequential(nn.Linear(2 * HID, HID), nn.ReLU(), nn.Linear(HID, 2))

    def forward(self, xs, xm):
        if self.family == "EARLY":
            return self.mlp(torch.cat([xs, xm], -1)).squeeze(-1)
        rs, ls = self.s(xs)
        rm, lm = self.m(xm)
        if self.family == "INTERMEDIATE":
            return self.head(torch.cat([rs, rm], -1)).squeeze(-1)
        if self.family in ("GATED", "MoE"):
            w = torch.softmax(self.gate(torch.cat([rs, rm], -1)), -1)
            return w[:, 0] * ls + w[:, 1] * lm
        return ls + lm            # LATE-AVG (equal weights, no fusion parameters)

    def embed(self, xs, xm):
        rs, _ = self.s(xs)
        rm, _ = self.m(xm)
        return torch.cat([rs, rm], -1)


def fit_net(family, Xs, Xm, y, tr, seed):
    torch.manual_seed(seed)
    net = Net(Xs.shape[1], Xm.shape[1], family)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xs, xm, yt = map(lambda a: torch.tensor(a, dtype=torch.float32), (Xs, Xm, y))
    idx = torch.tensor(tr)
    for _ in range(EPOCHS):
        net.train()
        for b in idx[torch.randperm(len(idx))].split(128):
            opt.zero_grad()
            lossf(net(xs[b], xm[b]), yt[b]).backward()
            opt.step()
    net.eval()
    return net, xs, xm


# ---------------------------------------------------------------- run
def run_benchmark(cohort: str = COH, verbose: bool = True):
    """Run every arm on identical folds. Returns (df, auroc, auprc, meta).

    `auroc` / `auprc` map arm -> list of per-fold scores, which `main()` does not need
    but the reproduction notebook does (forest-plot error bars, paired fold deltas).
    Extracted from `main()` verbatim so the notebook can IMPORT this rather than
    reimplement the fold loop -- a reimplementation is exactly how the two ad-hoc-feature
    errors documented in the module docstring got in.
    """
    Xm, Xs, y, Xtree = load(cohort)
    meta = dict(cohort=cohort, n=len(y), prevalence=float(y.mean()),
                seeds=SEEDS, epochs=EPOCHS, n_splits=5, n_repeats=2)
    if verbose:
        print(f"cohort {cohort}   n={len(y)}   prevalence={y.mean():.3f}")
        print(f"5x2 repeated stratified CV, {SEEDS} seeds/arm, identical folds for every arm")
        print(f"NOTE: all-negative accuracy on this cohort = {1-y.mean():.1%} -- "
              f"AUROC/AUPRC only.\n")

    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                          random_state=0).split(Xs, y))
    NEURAL = ["LATE-AVG", "EARLY", "INTERMEDIATE", "GATED", "MoE"]
    arms = NEURAL + ["LATE-STACK", "DEEP2GBM", "GBM", "LOGREG"]
    auroc = {a: [] for a in arms}
    auprc = {a: [] for a in arms}

    for k, (tr, te) in enumerate(splits, 1):
        sc_s = StandardScaler().fit(Xs[tr])
        sc_m = StandardScaler().fit(Xm[tr])
        Zs, Zm = sc_s.transform(Xs), sc_m.transform(Xm)

        for fam in NEURAL:
            ps = []
            for s in range(SEEDS):
                net, xs, xm = fit_net(fam, Zs, Zm, y, tr, s)
                with torch.no_grad():
                    ps.append(net(xs[te], xm[te]).numpy())
            p = np.mean(ps, 0)
            auroc[fam].append(roc_auc_score(y[te], p))
            auprc[fam].append(average_precision_score(y[te], p))

        # LATE-STACK and DEEP2GBM reuse the LATE-AVG towers, so the comparison is about
        # the COMBINER, not about tower quality.
        net, xs, xm = fit_net("LATE-AVG", Zs, Zm, y, tr, 0)
        with torch.no_grad():
            _, ls_tr = net.s(xs[tr]); _, lm_tr = net.m(xm[tr])
            _, ls_te = net.s(xs[te]); _, lm_te = net.m(xm[te])
            Etr, Ete = net.embed(xs[tr], xm[tr]).numpy(), net.embed(xs[te], xm[te]).numpy()
        Ztr = np.column_stack([ls_tr.numpy(), lm_tr.numpy()])
        Zte = np.column_stack([ls_te.numpy(), lm_te.numpy()])
        p = LogisticRegression(max_iter=1000).fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
        auroc["LATE-STACK"].append(roc_auc_score(y[te], p))
        auprc["LATE-STACK"].append(average_precision_score(y[te], p))

        p = HistGradientBoostingClassifier(random_state=42).fit(
            Etr, y[tr]).predict_proba(Ete)[:, 1]
        auroc["DEEP2GBM"].append(roc_auc_score(y[te], p))
        auprc["DEEP2GBM"].append(average_precision_score(y[te], p))

        p = HistGradientBoostingClassifier(random_state=42).fit(
            Xtree[tr], y[tr]).predict_proba(Xtree[te])[:, 1]
        auroc["GBM"].append(roc_auc_score(y[te], p))
        auprc["GBM"].append(average_precision_score(y[te], p))

        sc_t = StandardScaler().fit(Xtree[tr])
        p = LogisticRegression(max_iter=2000).fit(
            sc_t.transform(Xtree[tr]), y[tr]).predict_proba(sc_t.transform(Xtree[te]))[:, 1]
        auroc["LOGREG"].append(roc_auc_score(y[te], p))
        auprc["LOGREG"].append(average_precision_score(y[te], p))

        if verbose:
            print(f"  fold {k:2d}/10 done", flush=True)

    rows = [dict(family=a, auroc=np.mean(auroc[a]), auroc_sd=np.std(auroc[a]),
                 auprc=np.mean(auprc[a])) for a in arms]
    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    return df, auroc, auprc, meta


def main():
    df, auroc, auprc, meta = run_benchmark()
    best = df.iloc[0]

    print("\n" + "=" * 74)
    print(f"{'family':<14}{'AUROC':>16}{'AUPRC':>9}   vs best (2*SD rule)")
    print("-" * 74)
    for _, r in df.iterrows():
        a, b = np.array(auroc[r.family]), np.array(auroc[best.family])
        d, sd = a.mean() - b.mean(), np.std(np.concatenate([a, b]))
        tag = "best" if r.family == best.family else (
            "tie" if abs(d) <= 2 * sd else "DIFFERS")
        print(f"{r.family:<14}{r.auroc:>8.3f} +/- {r.auroc_sd:.3f}{r.auprc:>9.3f}   "
              f"{d:+.3f}  {tag}")
    print("=" * 74)
    df.to_csv(Path(__file__).parent / "published_families_benchmark.csv", index=False)
    print("\nAll arms share cohort, folds, features and budget, so differences are")
    print("attributable to architecture. Cross-DATASET AUROC comparisons with published")
    print("numbers are NOT valid and are deliberately not made.")


if __name__ == "__main__":
    main()
