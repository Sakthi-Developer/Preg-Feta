"""Wall-clock cost of the ACTUAL proposed models, not simplified stand-ins.

The eight-family benchmark deliberately gives every arm an identical generic tower so
that only the fusion mechanism varies. That is the right design for attributing an
ACCURACY difference to architecture -- but it means its runtime figures are not the cost
of this project's real models.

This script times the real ones as the repository actually configures them:
FETATransformer, PREGNet, HybridFusionPredictor(gated=True) and EnsemblePredictor,
trained through utils.training.Trainer on the standard PregnancyDataset loaders, at the
repository's default 80 epochs. Tree baselines are timed on the same split for reference.

Everything is timed on ONE train/val/test split of gen_hard, on CPU, at defaults.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from catboost import CatBoostClassifier                       # noqa: E402
from data.graph_builder import PatientGraphBuilder            # noqa: E402
from models.ensemble import EnsemblePredictor                 # noqa: E402
from models.feta_transformer import FETATransformer           # noqa: E402
from models.hybrid_fusion import HybridFusionPredictor        # noqa: E402
from models.preg_net import PREGNet                           # noqa: E402
from utils.training import Trainer, compute_class_weights     # noqa: E402

import cross_validate as CVMOD                                # noqa: E402

COH = ROOT / "data" / "gen_hard"
EPOCHS, BATCH, SEED = 80, 32, 42


def main():
    pat = pd.read_csv(COH / "patients.csv")
    scans = pd.read_csv(COH / "scans.csv")
    ids = pat["patient_id"].tolist()
    y = pat.set_index("patient_id")["label"]

    tr, te = train_test_split(ids, test_size=0.2, random_state=SEED,
                              stratify=y.loc[ids])
    tr, va = train_test_split(tr, test_size=0.2, random_state=SEED,
                              stratify=y.loc[tr])
    loaders, _ = CVMOD.build_loaders(pat, scans, tr, va, te, BATCH)
    train_labels = y.loc[tr].values
    device = torch.device("cpu")

    print(f"cohort gen_hard   train {len(tr)} / val {len(va)} / test {len(te)}")
    print(f"{EPOCHS} epochs, batch {BATCH}, CPU, repository defaults\n")

    specs = [
        ("FETA-Transformer", lambda: FETATransformer(), "feta", False),
        ("PREG-Net", lambda: PREGNet(), "preg", True),
        ("Gated hybrid", lambda: HybridFusionPredictor(gated=True), "ensemble", True),
        ("Ensemble (late fusion)", lambda: EnsemblePredictor(), "ensemble", True),
    ]

    rows = []
    for name, make, mode, needs_graph in specs:
        torch.manual_seed(SEED)
        model = make()
        gb = PatientGraphBuilder() if needs_graph else None
        ckpt = ROOT / "results" / "_runtime_tmp" / name.replace(" ", "_")
        ckpt.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model, train_loader=loaders["train"], val_loader=loaders["val"],
            lr=1e-3, weight_decay=1e-4,
            pos_weight=compute_class_weights(train_labels),
            patience=10_000,                      # disable early stop: time all epochs
            checkpoint_dir=str(ckpt), device=device,
            collate_mode=mode, graph_builder=gb,
            aux_loss_weight=0.3 if name == "Gated hybrid" else 0.0,
        )
        t0 = time.perf_counter()
        trainer.fit(EPOCHS)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics, preds = CVMOD.evaluate_deep_model(trainer, loaders["test"])
        pred_s = time.perf_counter() - t0

        n_par = sum(p.numel() for p in model.parameters())
        rows.append(dict(model=name, kind="proposed (real)", fit_single=fit_s,
                         pred_ms=pred_s * 1000, per_patient_ms=pred_s * 1000 / len(te),
                         auroc=metrics.get("auroc", float("nan")), params=n_par))
        print(f"  {name:24s} fit {fit_s:8.1f}s   predict {pred_s*1000:7.1f}ms   "
              f"AUROC {metrics.get('auroc', float('nan')):.3f}   params {n_par:,}",
              flush=True)

    # ---- tree / linear reference on the same split -------------------------
    from scripts.scaling_sweep import build_feature_table_fast
    ft = build_feature_table_fast(pat, scans).set_index("patient_id")
    X = ft.drop(columns=["label"]).astype(float)
    Xtr, Xte = X.loc[tr].values, X.loc[te].values
    ytr, yte = y.loc[tr].values, y.loc[te].values

    for name, make in [("CatBoost", lambda: CatBoostClassifier(
                            random_state=SEED, verbose=0, allow_writing_files=False)),
                       ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(
                            random_state=SEED)),
                       ("LogisticRegression", lambda: LogisticRegression(max_iter=2000))]:
        A, B = (Xtr, Xte)
        if name == "LogisticRegression":
            sc = StandardScaler().fit(Xtr)
            A, B = sc.transform(Xtr), sc.transform(Xte)
        m = make()
        t0 = time.perf_counter(); m.fit(A, ytr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(B)[:, 1]
        pred_s = time.perf_counter() - t0
        rows.append(dict(model=name, kind="baseline", fit_single=fit_s,
                         pred_ms=pred_s * 1000, per_patient_ms=pred_s * 1000 / len(te),
                         auroc=roc_auc_score(yte, p), params=np.nan))
        print(f"  {name:24s} fit {fit_s:8.3f}s   predict {pred_s*1000:7.1f}ms   "
              f"AUROC {roc_auc_score(yte, p):.3f}")

    df = pd.DataFrame(rows).sort_values("fit_single")
    df["slower_than_fastest"] = df.fit_single / df.fit_single.min()
    df.to_csv(Path(__file__).parent / "runtime_real_models.csv", index=False)

    print("\n" + "=" * 92)
    print(f"{'model':<26}{'fit (s)':>10}{'predict (ms)':>14}{'/patient (ms)':>15}"
          f"{'AUROC':>8}{'x slower':>10}")
    print("-" * 92)
    for r in df.itertuples():
        print(f"{r.model:<26}{r.fit_single:>10.2f}{r.pred_ms:>14.1f}"
              f"{r.per_patient_ms:>15.3f}{r.auroc:>8.3f}{r.slower_than_fastest:>9.0f}x")
    print("=" * 92)
    real = df[df.kind == "proposed (real)"]
    base = df[df.kind == "baseline"]
    print(f"\nfastest baseline      {base.fit_single.min():.3f}s")
    print(f"slowest real model    {real.fit_single.max():.1f}s "
          f"({real.fit_single.max()/base.fit_single.min():.0f}x slower)")
    print(f"AUROC spread overall  {df.auroc.max() - df.auroc.min():.3f}")
    print("\nNOTE single split, CPU, 80 epochs, early stopping disabled so that every "
          "model is timed over the same number of passes.")


if __name__ == "__main__":
    main()
