"""Run the eight-family benchmark on data/generated_dual -- the PUBLISHED Kaggle v2
cohort -- so the headline comparison exists on openly available data as well.

Same script, same folds, same budget as the gen_hard run: only the cohort changes.
Expect saturation here (every arm ~0.96-0.99); that is the point of reporting it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))

from published_families_benchmark import run_benchmark  # noqa: E402

df, auroc, auprc, meta = run_benchmark("data/generated_dual")

rows = []
for fam in auroc:
    a = np.array(auroc[fam])
    rows.append(dict(family=fam, auroc=a.mean(), auroc_sd=a.std(),
                     auprc=np.mean(auprc[fam])))
out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
out.to_csv(STUDY / "families_generated_dual.csv", index=False)

best = out.iloc[0]
print("\n" + "=" * 74)
print(f"{'family':<14}{'AUROC':>16}{'AUPRC':>9}   vs best")
print("-" * 74)
for r in out.itertuples():
    d = r.auroc - best.auroc
    sd = np.std(np.concatenate([auroc[r.family], auroc[best.family]]))
    tag = "best" if r.family == best.family else ("tie" if abs(d) <= 2 * sd else "DIFFERS")
    print(f"{r.family:<14}{r.auroc:>8.3f} +/- {r.auroc_sd:.3f}{r.auprc:>9.3f}   "
          f"{d:+.3f}  {tag}")
print("=" * 74)
print(f"spread {out.auroc.max() - out.auroc.min():.3f}   "
      f"smallest arm SD {out.auroc_sd.min():.3f}")
print(f"cohort {meta['cohort']}  n={meta['n']}  prevalence={meta['prevalence']:.3f}")
