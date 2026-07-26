"""Gate-headroom probe.

Answers, from per-fold predictions that already exist, two questions:
  1. CEILING    -- how much could ANY per-patient gate gain over a logit average?
                   (oracle hard gate: trust whichever tower is closer to the truth)
  2. LEARNABLE  -- can "which tower is better here" be predicted from the patient's
                   own features? A gate only realises the ceiling if this is > 0.5.

If (2) sits at chance, the gated architecture cannot win in this regime and we should
not spend a CV run on it.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[0]
REPO = Path(r"C:\Users\sakth\OneDrive\Desktop\Preg-Feta")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load(run_dir, model, fold):
    f = REPO / run_dir / model / f"fold_{fold}" / "test_predictions.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).set_index("patient_id")


def static_features(patients_csv, scans_csv, ids):
    pat = pd.read_csv(REPO / patients_csv).set_index("patient_id")
    sc = pd.read_csv(REPO / scans_csv)
    num = sc.select_dtypes("number")
    agg = num.groupby(sc["patient_id"]).agg(["mean", "std", "last", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    X = pat.join(agg, how="left")
    X = X.drop(columns=[c for c in X.columns if "label" in c.lower()], errors="ignore")
    X = X.select_dtypes("number")
    return X.reindex(ids)


def main(run_dir, patients_csv, scans_csv, folds=5):
    rows = []
    for k in range(1, folds + 1):
        f, p = load(run_dir, "feta", k), load(run_dir, "preg", k)
        if f is None or p is None:
            print(f"fold {k}: missing predictions, skipped")
            continue
        ids = f.index.intersection(p.index)
        y = f.loc[ids, "label"].values
        lf, lp = logit(f.loc[ids, "probability"].values), logit(p.loc[ids, "probability"].values)

        avg = (lf + lp) / 2
        # oracle hard gate: per patient, take the logit that ranks the patient correctly
        oracle = np.where(y == 1, np.maximum(lf, lp), np.minimum(lf, lp))

        a_f, a_p = roc_auc_score(y, lf), roc_auc_score(y, lp)
        a_avg, a_or = roc_auc_score(y, avg), roc_auc_score(y, oracle)

        # learnability: is "feta closer to truth than preg" predictable from features?
        err_f = np.abs(y - f.loc[ids, "probability"].values)
        err_p = np.abs(y - p.loc[ids, "probability"].values)
        prefer_feta = (err_f < err_p).astype(int)
        X = static_features(patients_csv, scans_csv, ids)
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
        oof = cross_val_predict(clf, X, prefer_feta, cv=3, method="predict_proba")[:, 1]
        a_learn = roc_auc_score(prefer_feta, oof)

        rows.append(dict(fold=k, feta=a_f, preg=a_p, avg=a_avg, oracle=a_or,
                         ceiling=a_or - a_avg, learnable=a_learn,
                         prefer_feta_rate=prefer_feta.mean(),
                         logit_corr=np.corrcoef(lf, lp)[0, 1]))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable folds")
        return
    print(df.round(3).to_string(index=False))
    m = df.mean(numeric_only=True)
    print("\n" + "=" * 62)
    print(f"  FETA {m.feta:.3f} | PREG {m.preg:.3f} | logit-avg {m.avg:.3f}")
    print(f"  tower logit correlation : {m.logit_corr:.3f}  (low = complementary)")
    print(f"  ORACLE gate             : {m.oracle:.3f}   ceiling over avg = +{m.ceiling:.3f}")
    print(f"  LEARNABILITY of gate    : {m.learnable:.3f} AUROC  (0.50 = hopeless)")
    print("=" * 62)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
