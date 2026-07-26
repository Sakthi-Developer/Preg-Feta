"""Do the paper's conclusions depend on WHICH gradient-boosting library we used?

The manuscript's tree baseline is scikit-learn's HistGradientBoosting. A reviewer can
fairly ask whether a different implementation -- XGBoost, LightGBM, or the CatBoost
used by Liu et al. -- would have beaten the fusion architectures rather than tying
them. This runs all four on the SAME cohort, the SAME folds and the SAME feature
table as the eight-family benchmark, so the only thing varying is the library.

Features come from the repository's build_feature_table_fast, reached through
published_families_benchmark.load(). Ad-hoc features understate the tree by ~0.019
and previously manufactured a false win; that path is deliberately not available here.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))

from published_families_benchmark import COH, load  # noqa: E402

from catboost import CatBoostClassifier          # noqa: E402
from lightgbm import LGBMClassifier              # noqa: E402
from xgboost import XGBClassifier                # noqa: E402

SEED = 42


def models():
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss",
                                 verbosity=0),
        "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=SEED, verbose=0,
                                       allow_writing_files=False),
    }


def main():
    _, Xs, y, Xtree = load(COH)
    print(f"cohort {COH}   n={len(y)}   prevalence={y.mean():.3f}")
    print(f"features: {Xtree.shape[1]} (build_feature_table_fast)")
    print("5x2 repeated stratified CV, identical folds for every library\n")

    # Identical splits to the eight-family benchmark (same seed, same generator).
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                          random_state=0).split(Xs, y))
    auroc = {k: [] for k in models()}
    auprc = {k: [] for k in models()}

    for k, (tr, te) in enumerate(splits, 1):
        for name, m in models().items():
            m.fit(Xtree[tr], y[tr])
            p = m.predict_proba(Xtree[te])[:, 1]
            auroc[name].append(roc_auc_score(y[te], p))
            auprc[name].append(average_precision_score(y[te], p))
        print(f"  fold {k:2d}/10 done", flush=True)

    rows = [dict(model=n, auroc=np.mean(auroc[n]), auroc_sd=np.std(auroc[n]),
                 auprc=np.mean(auprc[n])) for n in auroc]
    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    df.to_csv(STUDY / "tree_implementations.csv", index=False)

    best = df.iloc[0]
    print("\n" + "=" * 70)
    print(f"{'library':<24}{'AUROC':>16}{'AUPRC':>9}   vs best")
    print("-" * 70)
    for r in df.itertuples():
        a, b = np.array(auroc[r.model]), np.array(auroc[best.model])
        d = a.mean() - b.mean()
        sd = np.std(np.concatenate([a, b]))
        tag = "best" if r.model == best.model else (
            "tie" if abs(d) <= 2 * sd else "DIFFERS")
        print(f"{r.model:<24}{r.auroc:>8.3f} +/- {r.auroc_sd:.3f}{r.auprc:>9.3f}"
              f"   {d:+.3f}  {tag}")
    print("=" * 70)
    print(f"spread across libraries: {df.auroc.max() - df.auroc.min():.3f}")
    print(f"best tree ({best.model}): {best.auroc:.3f}")
    print("\nReference from the eight-family benchmark on this cohort:")
    print("  LATE-AVG (plain average, no learned fusion)  0.840")
    print("  GATED / MoE (gated fusion)                   0.827")
    d = best.auroc - 0.840
    print(f"\n  best tree - plain average = {d:+.3f}")


if __name__ == "__main__":
    main()
