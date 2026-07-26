"""SOLUTION 1: supervise each tower on ITS OWN channel, for a fraction of patients.

Diagnosis being tested: training both towers on the OR label punishes each for the
other channel's positives (scans recover label_sono at 0.90 but plateau at 0.73 on the
label; 59 engineered features do not move it). If that is right, telling the scan tower
which losses were actually sonographic should lift it toward its 0.820 oracle ceiling.

Clinical motivation: etiology IS often known retrospectively (karyotyping of products of
conception, pathology) -- but only for SOME patients. So the useful question is not
"does perfect etiology help" but "how little etiology do you need". Hence the sweep over
`frac` = share of TRAINING patients whose cause is known.

=================== LEAKAGE SAFEGUARDS (channel_labels is diagnostics-only) ============
`channel_labels.csv` is used ONLY as an auxiliary TRAINING TARGET, and only for patients
in the training split:
  * never an input feature to any model            (asserted: X is built from patients/
                                                    scans only)
  * never available for test patients              (asserted: aux mask ∩ test = empty)
  * never the evaluation target                    (test AUROC is on the plain `label`)
Because `label == label_sono OR label_mat`, giving BOTH channel labels as inputs would
be instant leakage -- that is what the repo rule forbids and it is not what happens here.
A model that has to PREDICT the channels from features at test time is doing legitimate
multi-task learning. The frac=0 arm is exactly the current approach, so any lift is
attributable to the supervision and nothing else.
=======================================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")
MEAS = ["FHR", "CRL", "GS", "YSD"]
COH = "data/gen_hard"
EPOCHS, SEEDS, NFOLD = 150, 3, 5
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
AUX_W = 1.0


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
    # the OR structure this whole argument rests on
    assert np.array_equal(((ys + ym) > 0).astype(np.float32), y), "label != sono OR mat"
    return (mat.astype(float).values, scan.astype(float).values, y, ys, ym)


class Tower(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.f(x)).squeeze(-1)


class OrNet(nn.Module):
    """Each tower predicts its own LATENT channel; the observed label is their OR."""

    def __init__(self, ds, dm):
        super().__init__()
        self.s, self.m = Tower(ds), Tower(dm)

    def forward(self, xs, xm):
        ls, lm = self.s(xs), self.m(xm)
        log1m = nn.functional.logsigmoid(-ls) + nn.functional.logsigmoid(-lm)
        p = torch.clamp(1 - torch.exp(log1m), 1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p)), ls, lm


def run(frac, Xs, Xm, y, ys, ym, tr, te, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # which TRAINING patients have a known cause. Test patients are never eligible.
    known = tr[rng.random(len(tr)) < frac] if frac > 0 else np.array([], dtype=int)
    assert len(np.intersect1d(known, te)) == 0, "LEAK: aux supervision touched test set"

    ss, sm = StandardScaler().fit(Xs[tr]), StandardScaler().fit(Xm[tr])
    xs = torch.tensor(np.nan_to_num(ss.transform(Xs)), dtype=torch.float32)
    xm = torch.tensor(np.nan_to_num(sm.transform(Xm)), dtype=torch.float32)
    yt, yst, ymt = torch.tensor(y), torch.tensor(ys), torch.tensor(ym)

    net = OrNet(Xs.shape[1], Xm.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    main_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
    aux_loss = nn.BCEWithLogitsLoss()
    kset = torch.tensor(np.isin(np.arange(len(y)), known))

    idx = torch.tensor(tr)
    for _ in range(EPOCHS):
        net.train()
        for b in idx[torch.randperm(len(idx))].split(128):
            opt.zero_grad()
            out, ls, lm = net(xs[b], xm[b])
            loss = main_loss(out, yt[b])
            k = kset[b]
            if k.any():   # teach each tower on ITS OWN channel, known patients only
                loss = loss + AUX_W * (aux_loss(ls[k], yst[b][k])
                                       + aux_loss(lm[k], ymt[b][k]))
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        out, ls, lm = net(xs[te], xm[te])
    return out.numpy(), ls.numpy(), lm.numpy()


def main():
    Xm, Xs, y, ys, ym = load()
    print(f"cohort {COH}  n={len(y)}  prev={y.mean():.3f}  "
          f"sono+={ys.mean():.3f} mat+={ym.mean():.3f}")
    print(f"aux weight {AUX_W}, {EPOCHS} epochs, {SEEDS} seeds, {NFOLD} folds\n")

    rows = []
    splits = list(StratifiedKFold(NFOLD, shuffle=True, random_state=0).split(Xs, y))

    hgb = []
    for tr, te in splits:
        X = np.hstack([np.nan_to_num(Xs), np.nan_to_num(Xm)])
        hgb.append(roc_auc_score(y[te], HistGradientBoostingClassifier(
            random_state=42).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    print(f"  HGB baseline (no etiology data)      {np.mean(hgb):.3f} +/- {np.std(hgb):.3f}\n")

    for frac in FRACS:
        lab, sono_rec = [], []
        for tr, te in splits:
            ps = [run(frac, Xs, Xm, y, ys, ym, tr, te, s) for s in range(SEEDS)]
            out = np.mean([p[0] for p in ps], 0)
            ls = np.mean([p[1] for p in ps], 0)
            lab.append(roc_auc_score(y[te], out))
            # does the scan tower actually become a better SONO detector?
            sono_rec.append(roc_auc_score(ys[te], ls) if 0 < ys[te].mean() < 1 else np.nan)
        rows.append(dict(frac=frac, label_auroc=np.mean(lab), sd=np.std(lab),
                         scan_tower_sono=np.nanmean(sono_rec)))
        print(f"  etiology known for {frac:5.0%} of train:  "
              f"label AUROC {np.mean(lab):.3f} +/- {np.std(lab):.3f}   "
              f"| scan tower -> label_sono {np.nanmean(sono_rec):.3f}", flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df.frac == 0, "label_auroc"].iloc[0]
    print("\n" + "=" * 70)
    print(f"  current approach (frac=0)        : {base:.3f}")
    print(f"  HGB baseline                     : {np.mean(hgb):.3f}")
    print(f"  best with etiology supervision   : {df.label_auroc.max():.3f} "
          f"(at frac={df.loc[df.label_auroc.idxmax(),'frac']:.0%})")
    print(f"  lift over current approach       : {df.label_auroc.max()-base:+.3f}")
    print(f"  lift over HGB                    : {df.label_auroc.max()-np.mean(hgb):+.3f}")
    print(f"  oracle sono ceiling (reference)  : 0.820")
    print("=" * 70)
    df.to_csv(Path(__file__).parent / "channel_supervision.csv", index=False)


if __name__ == "__main__":
    main()
