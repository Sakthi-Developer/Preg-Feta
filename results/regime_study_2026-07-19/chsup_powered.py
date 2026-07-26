"""POWERED re-test of channel supervision, with a shuffled-label control.

The 5-fold pilot gave +0.010 (0.835 -> 0.845), monotone and saturating at 25% -- but the
fold SD was 0.02, so it did not clear the repo's 2*SD rule, and the pilot's HGB baseline
used ad-hoc features (0.826) instead of the repo's own (0.836), inflating the gap.

Fixed here:
  * 5x2 repeated CV -> 10 estimates, so the decision rule has power
  * HGB gets the repo's build_feature_table_fast, the baseline analyze_prereg reports
  * SHUFFLED-CHANNEL CONTROL: identical training, but label_sono/label_mat are permuted
    across patients. Same extra loss term, same gradients, ZERO real etiology info.
    If shuffled matches real, the "lift" is regularisation, not information. This is the
    control that decides whether the idea is real.

Leakage safeguards unchanged: channel labels are a TRAINING-only auxiliary target on
training-split patients, never an input, never at test, never the eval target.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")
sys.path.insert(0, str(REPO))
MEAS = ["FHR", "CRL", "GS", "YSD"]
COH = "data/gen_hard"
EPOCHS, SEEDS = 150, 3
ARMS = [("frac=0.00 (current)", 0.0, False),
        ("frac=0.25 real", 0.25, False),
        ("frac=1.00 real", 1.0, False),
        ("frac=1.00 SHUFFLED", 1.0, True)]


def load():
    pat = pd.read_csv(REPO / COH / "patients.csv")
    sc = pd.read_csv(REPO / COH / "scans.csv").sort_values(
        ["patient_id", "gestational_age_weeks"])
    ch = pd.read_csv(REPO / COH / "channel_labels.csv").set_index("patient_id")
    ids = pat["patient_id"]
    y = pat["label"].values.astype(np.float32)

    mat = pat.drop(columns=["patient_id", "label"])
    cat = [c for c in mat.columns if not pd.api.types.is_numeric_dtype(mat[c])]
    mat = pd.get_dummies(mat, columns=cat, dummy_na=True) if cat else mat

    sc["visit"] = sc.groupby("patient_id").cumcount()
    flat = sc.pivot_table(index="patient_id", columns="visit",
                          values=MEAS + ["gestational_age_weeks"])
    flat.columns = [f"{a}_v{b}" for a, b in flat.columns]
    g = sc.groupby("patient_id")
    for m in MEAS:
        flat[f"{m}_mean"], flat[f"{m}_last"] = g[m].mean(), g[m].last()
    flat["n"] = g.size()
    scan = flat.reindex(ids).reset_index(drop=True)

    ys = ch.reindex(ids)["label_sono"].values.astype(np.float32)
    ym = ch.reindex(ids)["label_mat"].values.astype(np.float32)
    assert np.array_equal(((ys + ym) > 0).astype(np.float32), y), "label != sono OR mat"

    from scripts.scaling_sweep import build_feature_table_fast
    ft = build_feature_table_fast(pat, pd.read_csv(REPO / COH / "scans.csv"))
    assert np.array_equal(ft["label"].values.astype(np.float32), y), "row order mismatch"
    Xhgb = ft.drop(columns=["patient_id", "label"]).astype(float).values
    return mat.astype(float).values, scan.astype(float).values, y, ys, ym, Xhgb


class Tower(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.f(x)).squeeze(-1)


class OrNet(nn.Module):
    def __init__(self, ds, dm):
        super().__init__()
        self.s, self.m = Tower(ds), Tower(dm)

    def forward(self, xs, xm):
        ls, lm = self.s(xs), self.m(xm)
        log1m = nn.functional.logsigmoid(-ls) + nn.functional.logsigmoid(-lm)
        p = torch.clamp(1 - torch.exp(log1m), 1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p)), ls, lm


def run(frac, shuffled, Xs, Xm, y, ys, ym, tr, te, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if shuffled:   # destroy the information, keep the loss term identical
        perm = rng.permutation(len(y))
        ys, ym = ys[perm], ym[perm]

    known = tr[rng.random(len(tr)) < frac] if frac > 0 else np.array([], dtype=int)
    assert len(np.intersect1d(known, te)) == 0, "LEAK: aux supervision touched test"

    ss, sm = StandardScaler().fit(Xs[tr]), StandardScaler().fit(Xm[tr])
    xs = torch.tensor(np.nan_to_num(ss.transform(Xs)), dtype=torch.float32)
    xm = torch.tensor(np.nan_to_num(sm.transform(Xm)), dtype=torch.float32)
    yt, yst, ymt = torch.tensor(y), torch.tensor(ys), torch.tensor(ym)

    net = OrNet(Xs.shape[1], Xm.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    mainl, auxl = nn.BCEWithLogitsLoss(pos_weight=pw), nn.BCEWithLogitsLoss()
    kset = torch.tensor(np.isin(np.arange(len(y)), known))

    idx = torch.tensor(tr)
    for _ in range(EPOCHS):
        net.train()
        for b in idx[torch.randperm(len(idx))].split(128):
            opt.zero_grad()
            out, ls, lm = net(xs[b], xm[b])
            loss = mainl(out, yt[b])
            k = kset[b]
            if k.any():
                loss = loss + auxl(ls[k], yst[b][k]) + auxl(lm[k], ymt[b][k])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out, ls, _ = net(xs[te], xm[te])
    return out.numpy(), ls.numpy()


def main():
    Xm, Xs, y, ys, ym, Xhgb = load()
    print(f"cohort {COH}  n={len(y)}  prev={y.mean():.3f}   5x2 repeated CV, {SEEDS} seeds\n")
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                          random_state=0).split(Xs, y))

    hgb = [roc_auc_score(y[te], HistGradientBoostingClassifier(random_state=42)
                         .fit(Xhgb[tr], y[tr]).predict_proba(Xhgb[te])[:, 1])
           for tr, te in splits]
    print(f"  HGB (repo features)      {np.mean(hgb):.3f} +/- {np.std(hgb):.3f}\n")

    out = {}
    for name, frac, shuf in ARMS:
        lab, sono = [], []
        for tr, te in splits:
            ps = [run(frac, shuf, Xs, Xm, y, ys, ym, tr, te, s) for s in range(SEEDS)]
            lab.append(roc_auc_score(y[te], np.mean([p[0] for p in ps], 0)))
            sono.append(roc_auc_score(ys[te], np.mean([p[1] for p in ps], 0)))
        out[name] = np.array(lab)
        print(f"  {name:22s} label {np.mean(lab):.3f} +/- {np.std(lab):.3f} "
              f"| scan->sono {np.mean(sono):.3f}", flush=True)

    base = out["frac=0.00 (current)"]
    real = out["frac=1.00 real"]
    shuf = out["frac=1.00 SHUFFLED"]
    print("\n" + "=" * 72)
    def rule(a, b, tag):
        d, sd = a.mean() - b.mean(), np.std(np.concatenate([a, b]))
        w = int((a > b).sum())
        v = "WIN" if (d > 2 * sd and w >= 0.8 * len(a)) else "tie"
        print(f"  {tag:34s} {d:+.3f}   2*SD={2*sd:.3f}  sign {w}/{len(a)}  -> {v}")
    rule(real, base, "real supervision vs current")
    rule(real, np.array(hgb), "real supervision vs HGB")
    rule(real, shuf, "REAL vs SHUFFLED (the control)")
    print("\n  If REAL vs SHUFFLED is a tie, the lift is regularisation, not etiology.")
    print("=" * 72)


if __name__ == "__main__":
    main()
