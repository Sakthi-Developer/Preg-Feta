"""FEASIBILITY: does a NOISY-OR fusion head beat additive fusion?

Motivation: on gen_hard the label is generated as `label_sono OR label_mat`, but every
fusion tried so far (logit average, gated cross-attention, joint HGB) combines the two
towers ADDITIVELY. A tower trained on the OR label is punished for patients who lost
for the OTHER channel's reason -- which is why scan features plateau at 0.73 while
recovering label_sono at 0.90.

Noisy-OR combines them the way the etiology does:
    P(loss) = 1 - (1 - p_sono)(1 - p_mat)
Each tower predicts its OWN latent channel; only the OR is supervised. channel_labels
is NEVER read here -- the channels stay latent, so nothing leaks.

Towers are small MLPs so the comparison isolates the FUSION HEAD, not tower capacity:
every arm below uses identical tower architecture, identical features, identical
budget. Arms:
    OR      noisy-OR head            (structural OR)
    ADD     logit-sum head           (additive, same towers)
    CONCAT  MLP on concatenated reps (learned, same towers)
    HGB     joint tree on all features (the baseline that has won everything)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")
MEAS = ["FHR", "CRL", "GS", "YSD"]
EPOCHS, SEEDS = 150, 3


def features(coh):
    pat = pd.read_csv(REPO / coh / "patients.csv")
    sc = pd.read_csv(REPO / coh / "scans.csv").sort_values(
        ["patient_id", "gestational_age_weeks"])
    y = pat["label"].values.astype(np.float32)
    ids = pat["patient_id"]

    mat = pat.drop(columns=["patient_id", "label"])
    cat = [c for c in mat.columns if not pd.api.types.is_numeric_dtype(mat[c])]
    mat = pd.get_dummies(mat, columns=cat, dummy_na=True) if cat else mat

    sc["visit"] = sc.groupby("patient_id").cumcount()
    flat = sc.pivot_table(index="patient_id", columns="visit",
                          values=MEAS + ["gestational_age_weeks"])
    flat.columns = [f"{a}_v{b}" for a, b in flat.columns]
    g = sc.groupby("patient_id")
    for m in MEAS:
        flat[f"{m}_mean"] = g[m].mean()
        flat[f"{m}_last"] = g[m].last()
    flat["n"] = g.size()
    scan = flat.reindex(ids).reset_index(drop=True)
    return mat.astype(float).values, scan.astype(float).values, y


class Tower(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        r = self.f(x)
        return r, self.head(r).squeeze(-1)


class Fusion(nn.Module):
    def __init__(self, ds, dm, mode):
        super().__init__()
        self.s, self.m, self.mode = Tower(ds), Tower(dm), mode
        if mode == "concat":
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, xs, xm):
        rs, ls = self.s(xs)
        rm, lm = self.m(xm)
        if self.mode == "or":
            # log(1-p) summed = noisy-OR, computed in log space for stability
            log1m = nn.functional.logsigmoid(-ls) + nn.functional.logsigmoid(-lm)
            p = torch.clamp(1 - torch.exp(log1m), 1e-6, 1 - 1e-6)
            return torch.log(p / (1 - p))
        if self.mode == "add":
            return ls + lm
        return self.head(torch.cat([rs, rm], -1)).squeeze(-1)


def train_eval(mode, Xs, Xm, y, tr, te, seed):
    torch.manual_seed(seed)
    ss, sm = StandardScaler().fit(Xs[tr]), StandardScaler().fit(Xm[tr])
    xs = torch.tensor(np.nan_to_num(ss.transform(Xs)), dtype=torch.float32)
    xm = torch.tensor(np.nan_to_num(sm.transform(Xm)), dtype=torch.float32)
    yt = torch.tensor(y)
    net = Fusion(Xs.shape[1], Xm.shape[1], mode)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    idx = torch.tensor(tr)
    for _ in range(EPOCHS):
        net.train()
        perm = idx[torch.randperm(len(idx))]
        for b in perm.split(128):
            opt.zero_grad()
            lossf(net(xs[b], xm[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xs[te], xm[te]).numpy()


def repo_hgb_features(coh):
    """The repo's OWN tabular features, so the tree baseline is the one
    analyze_prereg.py reports (0.836 on gen_hard) -- not a weaker ad-hoc version.
    Using my own features here handicapped HGB by ~0.019, which would have turned a
    tie into a fake win."""
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.scaling_sweep import build_feature_table_fast
    f = build_feature_table_fast(pd.read_csv(REPO / coh / "patients.csv"),
                                 pd.read_csv(REPO / coh / "scans.csv"))
    y = f["label"].values.astype(np.float32)
    X = f.drop(columns=["patient_id", "label"]).astype(float).values
    return X, y


def main():
    # gen_hard is dual-etiology (label = sono OR mat) -- noisy-OR should help.
    # generated is SINGLE-etiology -- the OR prior is mis-specified there, so it should
    # NOT help. A win on both would mean my OR arm is just a better-regularised net,
    # and the mechanism story would be wrong.
    for coh in ["data/gen_hard", "data/generated"]:
        Xm, Xs, y = features(coh)
        Xhgb, yh = repo_hgb_features(coh)
        assert np.array_equal(y, yh), "label order mismatch between feature builders"
        print(f"\n{'='*58}\ncohort {coh}  n={len(y)}  prev={y.mean():.3f}  "
              f"scan_d={Xs.shape[1]} mat_d={Xm.shape[1]}")
        res = {k: [] for k in ["OR", "ADD", "CONCAT", "HGB"]}
        # 5x2 repeated CV: 10 estimates instead of 5, so the decision rule has power.
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
        for f, (tr, te) in enumerate(splitter.split(Xs, y)):
            res["HGB"].append(roc_auc_score(y[te], HistGradientBoostingClassifier(
                random_state=42).fit(Xhgb[tr], y[tr]).predict_proba(Xhgb[te])[:, 1]))
            for tag, mode in [("OR", "or"), ("ADD", "add"), ("CONCAT", "concat")]:
                p = np.mean([train_eval(mode, Xs, Xm, y, tr, te, s) for s in range(SEEDS)], 0)
                res[tag].append(roc_auc_score(y[te], p))
            print(f"  fold {f+1}: " + "  ".join(
                f"{k} {res[k][-1]:.3f}" for k in res), flush=True)

        print("\n" + "-" * 58)
        for k, v in res.items():
            print(f"  {k:7s} {np.mean(v):.3f} +/- {np.std(v):.3f}")
        print("-" * 58)
        # The repo's fixed decision rule: gap must exceed 2*SD AND agree in sign on
        # >=80% of folds. Anything less is a TIE, whatever the sign.
        o = np.array(res["OR"])
        for opp in ["ADD", "CONCAT", "HGB"]:
            b = np.array(res[opp])
            d, sd = o.mean() - b.mean(), np.std(np.concatenate([o, b]))
            wins = int((o > b).sum())
            verdict = "WIN" if (d > 2 * sd and wins >= 0.8 * len(o)) else "tie"
            print(f"  OR - {opp:6s} = {d:+.3f}   2*SD={2*sd:.3f}   "
                  f"sign {wins}/{len(o)}   -> {verdict}")


if __name__ == "__main__":
    main()
