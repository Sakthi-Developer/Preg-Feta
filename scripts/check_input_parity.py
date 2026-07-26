"""Assert every model actually READS every modality it is credited with.

Why this exists
---------------
From the introduction of the input-parity rule until 2026-07-16, every results table
in this repo credited FETA with "scans + maternal". It was scans-only. Its
`MaternalCrossAttention` used the pooled scan vector as its single key, and a softmax
over one key is identically 1.0, so the maternal query was computed and discarded. The
module even returned pre-softmax scores that varied with maternal, so the pathway
*looked* alive under inspection. Nothing caught it for weeks of runs.

Input parity is a property of the WIRING, not of the CLI flags. A model can be handed a
tensor and silently ignore it. The only honest check is: does the output actually depend
on the input?

Method
------
Two independent tests per (model, modality):

  1. GRADIENT  — d(logits)/d(input) must be non-zero. A dead pathway gives exactly 0.0.
  2. PERTURBATION — feeding a very different input must change the logits. Catches the
     case where a gradient exists but is numerically severed downstream.

Run this after ANY change to a model's forward signature or fusion wiring:

    .venv/Scripts/python.exe scripts/check_input_parity.py

Exits non-zero on failure, so it can gate a commit.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from data.graph_builder import PatientGraphBuilder
from models import FETATransformer, HybridFusionPredictor, PREGNet

B, T = 6, 5
FAILURES: list[str] = []


def _batch(requires_grad_maternal: bool = False, scale_maternal: float = 1.0):
    torch.manual_seed(7)
    return {
        "temporal_features": torch.randn(B, T, 4),
        "gestational_ages": torch.linspace(6.0, 12.0, T).unsqueeze(0).repeat(B, 1),
        "temporal_mask": torch.ones(B, T),
        "maternal_features": (
            torch.randn(B, 7, generator=torch.Generator().manual_seed(3)) * scale_maternal
        ).requires_grad_(requires_grad_maternal),
    }


def _report(model: str, modality: str, grad: float, delta: float) -> None:
    ok = grad > 0.0 and delta > 1e-6
    status = "PASS" if ok else "**FAIL**"
    print(f"  {status:8} {model:<22} reads {modality:<18} "
          f"grad={grad:.6g}  max|d_logit|={delta:.6g}")
    if not ok:
        why = "zero gradient" if grad == 0.0 else "output does not move"
        FAILURES.append(f"{model} does NOT read {modality} ({why})")


def check_feta() -> None:
    print("\nFETA-Transformer")
    m = FETATransformer().eval()

    b = _batch(requires_grad_maternal=True)
    m(**b)[0].sum().backward()
    grad = b["maternal_features"].grad.abs().sum().item()

    a = _batch(scale_maternal=1.0)
    c = _batch(scale_maternal=50.0)
    with torch.no_grad():
        delta = (m(**a)[0] - m(**c)[0]).abs().max().item()
    _report("FETA", "maternal", grad, delta)

    # scans
    b2 = _batch()
    b2["temporal_features"].requires_grad_(True)
    m(**b2)[0].sum().backward()
    g2 = b2["temporal_features"].grad.abs().sum().item()
    d2 = _batch()
    d2["temporal_features"] = d2["temporal_features"] * 50.0
    with torch.no_grad():
        delta2 = (m(**_batch())[0] - m(**d2)[0]).abs().max().item()
    _report("FETA", "scans", g2, delta2)


def check_preg() -> None:
    print("\nPREG-Net (full graph, i.e. NOT --clinical-only)")
    b = _batch()
    g = PatientGraphBuilder(max_scans=T).batch_to_graph(b)
    m = PREGNet().eval()

    nf = g["node_features"].clone().requires_grad_(True)
    m(nf, g["edge_index"], g["node_types"], g["node_mask"])[0].sum().backward()
    grad = nf.grad.abs().sum().item()
    with torch.no_grad():
        base = m(g["node_features"], g["edge_index"], g["node_types"], g["node_mask"])[0]
        pert = m(g["node_features"] * 50.0, g["edge_index"], g["node_types"],
                 g["node_mask"])[0]
    _report("PREG-Net", "graph nodes", grad, (base - pert).abs().max().item())


def check_hybrid() -> None:
    print("\nHybridFusionPredictor (gated)")
    m = HybridFusionPredictor(gated=True).eval()

    def fwd(batch, graph):
        return m(batch["temporal_features"], batch["gestational_ages"],
                 batch["temporal_mask"], batch["maternal_features"],
                 graph["node_features"], graph["edge_index"], graph["node_types"],
                 graph["node_mask"])

    b = _batch(requires_grad_maternal=True)
    g = PatientGraphBuilder(max_scans=T).batch_to_graph(b)
    out = fwd(b, g)

    # The FETA branch specifically must read maternal -- if only PREG-Net does, the
    # hybrid is PREG-Net plus a maternal-blind tower, which is the pre-fix state.
    out["feta_logits"].sum().backward(retain_graph=True)
    grad_feta_branch = b["maternal_features"].grad.abs().sum().item()

    a, aa = _batch(scale_maternal=1.0), _batch(scale_maternal=50.0)
    ga_ = PatientGraphBuilder(max_scans=T).batch_to_graph(a)
    gb_ = PatientGraphBuilder(max_scans=T).batch_to_graph(aa)
    with torch.no_grad():
        delta = (fwd(a, ga_)["feta_logits"] - fwd(aa, gb_)["feta_logits"]).abs().max().item()
    _report("Hybrid/FETA branch", "maternal", grad_feta_branch, delta)


def main() -> None:
    print("=" * 78)
    print("INPUT-PARITY CHECK -- does each model actually read what it is credited with?")
    print("=" * 78)
    check_feta()
    check_preg()
    check_hybrid()

    print("\n" + "=" * 78)
    if FAILURES:
        print("PARITY VIOLATED:")
        for f in FAILURES:
            print("  - " + f)
        print("\nAny results table crediting that model with that modality is MISLABELED.")
        sys.exit(1)
    print("All models read every modality they are credited with.")
    print("=" * 78)


if __name__ == "__main__":
    main()
