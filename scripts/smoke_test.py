"""Smoke-test the model/data wiring with tiny synthetic tensors."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.graph_builder import PatientGraphBuilder
from models import EnsemblePredictor, FETATransformer, PREGNet


def make_batch(batch_size: int = 2, max_scans: int = 5):
    return {
        "temporal_features": torch.randn(batch_size, max_scans, 4),
        "gestational_ages": torch.tensor(
            [
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [6.5, 8.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )[:batch_size],
        "temporal_mask": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )[:batch_size],
        "maternal_features": torch.randn(batch_size, 7),
    }


def main() -> None:
    torch.manual_seed(42)
    batch = make_batch()
    graph = PatientGraphBuilder(max_scans=5).batch_to_graph(batch)

    feta = FETATransformer()
    feta_logits, feta_attn = feta(**batch)
    assert feta_logits.shape == (2, 1)
    assert feta_attn["temporal_pooling_weights"].shape == (2, 5)

    preg = PREGNet()
    preg_logits, preg_attn = preg(
        graph["node_features"],
        graph["edge_index"],
        graph["node_types"],
        graph["node_mask"],
    )
    assert preg_logits.shape == (2, 1)
    assert preg_attn["node_importance"].shape[0] == 2

    ensemble = EnsemblePredictor()
    outputs = ensemble(
        temporal_features=batch["temporal_features"],
        gestational_ages=batch["gestational_ages"],
        temporal_mask=batch["temporal_mask"],
        maternal_features=batch["maternal_features"],
        node_features=graph["node_features"],
        edge_index=graph["edge_index"],
        node_types=graph["node_types"],
        node_mask=graph["node_mask"],
    )
    assert outputs["logits"].shape == (2, 1)

    print("Smoke test passed: FETA, PREG-Net, and ensemble forward paths work.")


if __name__ == "__main__":
    main()
