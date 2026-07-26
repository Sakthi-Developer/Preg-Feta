"""What does the extra architecture COST, given that it buys no accuracy?

Accuracy ties across every arm. Runtime does not. This measures wall-clock training
and inference time for each arm on identical folds and identical features, so the
accuracy-per-second trade-off can be stated with numbers rather than asserted.

FAIRNESS NOTES -- read before quoting these figures:
  * Timings are wall clock on ONE machine, CPU only, at library defaults. They are
    not a claim about the algorithms in general; a GPU, a tuned batch size, or
    fewer epochs would move the neural numbers substantially.
  * The neural arms are timed TWO ways: "as run in this study" (3 seeds averaged,
    150 epochs each, which is what produced the paper's AUROC) and "single fit"
    (1 seed), so the comparison is not inflated by our own seed-averaging choice.
    The single-fit column is the honest like-for-like against one tree fit.
  * Trees and linear models use all available cores by default; torch likewise.
    Thread counts are not equalised, because the point is what a practitioner
    actually experiences at default settings.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))

from published_families_benchmark import COH, SEEDS, Net, fit_net, load  # noqa: E402

from catboost import CatBoostClassifier   # noqa: E402
from lightgbm import LGBMClassifier       # noqa: E402
from xgboost import XGBClassifier         # noqa: E402

NEURAL = ["LATE-AVG", "EARLY", "INTERMEDIATE", "GATED"]
N_FOLDS = 5


def time_trees(Xtree, y, splits):
    defs = {
        "GBM (HistGradientBoosting)": lambda: HistGradientBoostingClassifier(random_state=42),
        "CatBoost": lambda: CatBoostClassifier(random_state=42, verbose=0,
                                               allow_writing_files=False),
        "XGBoost": lambda: XGBClassifier(random_state=42, eval_metric="logloss",
                                         verbosity=0),
        "LightGBM": lambda: LGBMClassifier(random_state=42, verbose=-1),
        "LOGREG": lambda: LogisticRegression(max_iter=2000),
    }
    out = {}
    for name, make in defs.items():
        fit_t, pred_t, auc = [], [], []
        for tr, te in splits:
            Xtr, Xte = Xtree[tr], Xtree[te]
            if name == "LOGREG":
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = make()
            t0 = time.perf_counter()
            m.fit(Xtr, y[tr])
            fit_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            p = m.predict_proba(Xte)[:, 1]
            pred_t.append(time.perf_counter() - t0)
            auc.append(roc_auc_score(y[te], p))
        out[name] = dict(fit_single=np.mean(fit_t), fit_study=np.mean(fit_t),
                         pred_ms=np.mean(pred_t) * 1000, auroc=np.mean(auc),
                         kind="tree/linear")
        print(f"  {name:28s} fit {np.mean(fit_t):7.3f}s  AUROC {np.mean(auc):.3f}",
              flush=True)
    return out


def time_neural(Zs, Zm, y, splits):
    out = {}
    for fam in NEURAL:
        single, pred_t, auc = [], [], []
        for tr, te in splits:
            t0 = time.perf_counter()
            net, xs, xm = fit_net(fam, Zs, Zm, y, tr, 0)
            single.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            with torch.no_grad():
                p = net(xs[te], xm[te]).numpy()
            pred_t.append(time.perf_counter() - t0)
            auc.append(roc_auc_score(y[te], p))
        m = np.mean(single)
        out[fam] = dict(fit_single=m, fit_study=m * SEEDS,
                        pred_ms=np.mean(pred_t) * 1000, auroc=np.mean(auc),
                        kind="neural")
        print(f"  {fam:28s} fit {m:7.3f}s (x{SEEDS} seeds = {m * SEEDS:6.1f}s)"
              f"  AUROC {np.mean(auc):.3f}", flush=True)
    return out


def time_two_stage(Zs, Zm, y, splits):
    """LATE-STACK and DEEP2GBM: tower training PLUS a second-stage combiner.

    Both reuse the LATE-AVG towers, so their cost is tower training + the combiner
    fitted on top. Timed end-to-end because that is what a practitioner pays.
    """
    out = {}
    for fam in ("LATE-STACK", "DEEP2GBM"):
        single, pred_t, auc = [], [], []
        for tr, te in splits:
            t0 = time.perf_counter()
            net, xs, xm = fit_net("LATE-AVG", Zs, Zm, y, tr, 0)
            with torch.no_grad():
                if fam == "LATE-STACK":
                    _, ls_tr = net.s(xs[tr]); _, lm_tr = net.m(xm[tr])
                    _, ls_te = net.s(xs[te]); _, lm_te = net.m(xm[te])
                    Atr = np.column_stack([ls_tr.numpy(), lm_tr.numpy()])
                    Ate = np.column_stack([ls_te.numpy(), lm_te.numpy()])
                else:
                    Atr = net.embed(xs[tr], xm[tr]).numpy()
                    Ate = net.embed(xs[te], xm[te]).numpy()
            comb = (LogisticRegression(max_iter=1000) if fam == "LATE-STACK"
                    else HistGradientBoostingClassifier(random_state=42))
            comb.fit(Atr, y[tr])
            single.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            p = comb.predict_proba(Ate)[:, 1]
            pred_t.append(time.perf_counter() - t0)
            auc.append(roc_auc_score(y[te], p))
        m = np.mean(single)
        out[fam] = dict(fit_single=m, fit_study=m, pred_ms=np.mean(pred_t) * 1000,
                        auroc=np.mean(auc), kind="neural+combiner")
        print(f"  {fam:28s} fit {m:7.3f}s  AUROC {np.mean(auc):.3f}", flush=True)
    return out


def main():
    Xm, Xs, y, Xtree = load(COH)
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True,
                                  random_state=0).split(Xs, y))
    print(f"cohort {COH}  n={len(y)}  {N_FOLDS}-fold, CPU, library defaults")
    print(f"torch threads={torch.get_num_threads()}   epochs=150 per neural fit\n")

    sc_s, sc_m = StandardScaler().fit(Xs), StandardScaler().fit(Xm)
    Zs, Zm = sc_s.transform(Xs), sc_m.transform(Xm)

    print("trees / linear:")
    res = time_trees(Xtree, y, splits)
    print("\nneural (single seed timed; study used 3 seeds):")
    res.update(time_neural(Zs, Zm, y, splits))
    print("\ntwo-stage published families (towers + second-stage combiner):")
    res.update(time_two_stage(Zs, Zm, y, splits))

    df = pd.DataFrame([dict(model=k, **v) for k, v in res.items()])
    fastest = df.fit_single.min()
    df["slower_than_fastest"] = df.fit_single / fastest
    df = df.sort_values("fit_single")
    df.to_csv(STUDY / "runtime_benchmark.csv", index=False)

    print("\n" + "=" * 88)
    print(f"{'model':<28}{'fit (s)':>9}{'as-run (s)':>12}{'predict(ms)':>13}"
          f"{'AUROC':>8}{'x slower':>10}")
    print("-" * 88)
    for r in df.itertuples():
        print(f"{r.model:<28}{r.fit_single:>9.3f}{r.fit_study:>12.1f}"
              f"{r.pred_ms:>13.2f}{r.auroc:>8.3f}{r.slower_than_fastest:>9.0f}x")
    print("=" * 88)

    tr = df[df.kind == "tree/linear"]
    nn = df[df.kind == "neural"]
    print(f"\nfastest tree/linear : {tr.fit_single.min():.3f}s")
    print(f"slowest neural      : {nn.fit_single.max():.3f}s "
          f"({nn.fit_single.max() / tr.fit_single.min():.0f}x slower, single fit)")
    print(f"best AUROC overall  : {df.loc[df.auroc.idxmax(), 'model']} "
          f"({df.auroc.max():.3f})")
    print(f"AUROC spread        : {df.auroc.max() - df.auroc.min():.3f}")


if __name__ == "__main__":
    main()
