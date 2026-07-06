"""
Knowledge-guided graph construction for PREG-Net.

Builds patient-level graphs where:
  - Nodes represent clinical variables (FHR, CRL, GS, YSD per time step
    + static maternal features)
  - Edges encode physiological relationships (knowledge-guided)
  - Temporal edges connect same variable across consecutive time steps

All implemented in pure PyTorch — no PyTorch Geometric dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Node type constants
# ──────────────────────────────────────────────────────────────────────────────

# Temporal node types (one per time step per variable)
NODE_FHR = 0
NODE_CRL = 1
NODE_GS = 2
NODE_YSD = 3

# Static maternal node types
NODE_AGE = 4
NODE_BMI = 5
NODE_PARITY = 6
NODE_GRAVIDITY = 7
NODE_PREV_LOSS = 8
NODE_CONCEPTION = 9
NODE_SINGLETON = 10

N_US_TYPES = 4          # FHR, CRL, GS, YSD
N_MATERNAL_TYPES = 7    # age, bmi, parity, gravidity, prev_loss, conception, singleton
N_NODE_TYPES = N_US_TYPES + N_MATERNAL_TYPES  # 11

# Edge type constants
EDGE_PHYSIOLOGICAL = 0
EDGE_TEMPORAL = 1
EDGE_MATERNAL_TO_US = 2


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge-guided graph edges
# ──────────────────────────────────────────────────────────────────────────────

# Physiological edges between US measurement types (within same time step)
# Based on research paper's Table 1:
#   FHR → CRL : cardiac output influences nutrient delivery and growth
#   YSD → GS  : yolk sac function supports gestational sac development
#   CRL → GS  : embryonic growth correlates with sac growth
INTRA_TIME_EDGES = [
    (NODE_FHR, NODE_CRL),
    (NODE_CRL, NODE_FHR),  # bidirectional
    (NODE_YSD, NODE_GS),
    (NODE_GS, NODE_YSD),   # bidirectional
    (NODE_CRL, NODE_GS),
    (NODE_GS, NODE_CRL),   # bidirectional
    (NODE_FHR, NODE_YSD),
    (NODE_YSD, NODE_FHR),  # bidirectional
]

# Maternal → US edges (static nodes influence all temporal US nodes)
# Age → FHR, CRL (maternal age affects embryonic development)
# BMI → All US (metabolic status modulates all parameters)
# Prior_loss → All US (history indicates underlying risk)
MATERNAL_TO_US_MAPPING = {
    NODE_AGE: [NODE_FHR, NODE_CRL, NODE_GS, NODE_YSD],
    NODE_BMI: [NODE_FHR, NODE_CRL, NODE_GS, NODE_YSD],
    NODE_PREV_LOSS: [NODE_FHR, NODE_CRL, NODE_GS, NODE_YSD],
    NODE_GRAVIDITY: [NODE_FHR, NODE_CRL],
    NODE_PARITY: [NODE_FHR, NODE_CRL],
    NODE_CONCEPTION: [NODE_FHR, NODE_CRL, NODE_GS],
}


def build_knowledge_graph(
    n_time_steps: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the knowledge-guided graph structure.

    The graph has:
      - n_time_steps * 4 temporal nodes (FHR, CRL, GS, YSD at each time step)
      - 7 static maternal nodes
      - Total = n_time_steps * 4 + 7 nodes

    Node indexing:
      [0 .. n_time_steps*4 - 1]  : temporal nodes (grouped by time step)
        time t: [t*4 + 0] = FHR_t, [t*4 + 1] = CRL_t,
                [t*4 + 2] = GS_t,  [t*4 + 3] = YSD_t
      [n_time_steps*4 .. n_time_steps*4 + 6] : maternal nodes

    Returns
    -------
    edge_index : (2, E) long tensor — COO format
    edge_type  : (E,) long tensor — 0=physiological, 1=temporal, 2=maternal_to_us
    """
    n_temporal = n_time_steps * N_US_TYPES
    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_types: List[int] = []

    # 1. Intra-time-step physiological edges
    for t in range(n_time_steps):
        offset = t * N_US_TYPES
        for src_type, dst_type in INTRA_TIME_EDGES:
            edges_src.append(offset + src_type)
            edges_dst.append(offset + dst_type)
            edge_types.append(EDGE_PHYSIOLOGICAL)

    # 2. Temporal edges (same variable across consecutive time steps)
    for var_type in range(N_US_TYPES):
        for t in range(n_time_steps - 1):
            src = t * N_US_TYPES + var_type
            dst = (t + 1) * N_US_TYPES + var_type
            # Forward
            edges_src.append(src)
            edges_dst.append(dst)
            edge_types.append(EDGE_TEMPORAL)
            # Backward
            edges_src.append(dst)
            edges_dst.append(src)
            edge_types.append(EDGE_TEMPORAL)

    # 3. Maternal → temporal US edges
    for mat_type, us_targets in MATERNAL_TO_US_MAPPING.items():
        mat_idx = n_temporal + (mat_type - N_US_TYPES)
        for t in range(n_time_steps):
            for us_type in us_targets:
                us_idx = t * N_US_TYPES + us_type
                edges_src.append(mat_idx)
                edges_dst.append(us_idx)
                edge_types.append(EDGE_MATERNAL_TO_US)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    return edge_index, edge_type


# ──────────────────────────────────────────────────────────────────────────────
# Per-patient graph builder
# ──────────────────────────────────────────────────────────────────────────────

class PatientGraphBuilder:
    """
    Converts a batch from PregnancyDataset into graph-structured data
    for PREG-Net.

    Each patient becomes a graph with:
      - Temporal nodes: FHR_t, CRL_t, GS_t, YSD_t for t=0..T-1
      - Maternal nodes: age, bmi, parity, gravidity, prev_loss, conception, singleton
      - Knowledge-guided edges shared across all patients
    """

    def __init__(self, max_scans: int = 5):
        self.max_scans = max_scans
        self.n_temporal = max_scans * N_US_TYPES
        self.n_maternal = N_MATERNAL_TYPES
        self.n_nodes = self.n_temporal + self.n_maternal

        # Pre-build shared graph structure
        self.edge_index, self.edge_type = build_knowledge_graph(max_scans)

        # Pre-build node type assignments
        node_types = []
        for t in range(max_scans):
            node_types.extend([NODE_FHR, NODE_CRL, NODE_GS, NODE_YSD])
        node_types.extend([
            NODE_AGE, NODE_BMI, NODE_PARITY, NODE_GRAVIDITY,
            NODE_PREV_LOSS, NODE_CONCEPTION, NODE_SINGLETON
        ])
        self.node_types = torch.tensor(node_types, dtype=torch.long)

    def batch_to_graph(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a collated batch dict into graph tensors for PREG-Net.

        Parameters
        ----------
        batch : dict from PregnancyDataset collate_fn

        Returns
        -------
        dict with:
            node_features : (B, N_nodes, 1) — scalar value per node
            edge_index    : (2, E) — shared graph structure
            edge_type     : (E,) — edge type labels
            node_types    : (N_nodes,) — node type indices
            node_mask     : (B, N_nodes) — 1.0 for valid nodes, 0.0 for padded
        """
        B = batch['temporal_features'].shape[0]
        T = self.max_scans

        # Temporal features: (B, T, 4) → (B, T*4)
        temporal = batch['temporal_features']  # (B, T, 4)
        temporal_flat = temporal.reshape(B, T * N_US_TYPES)  # (B, T*4)

        # Maternal features: (B, 7)
        maternal = batch['maternal_features']  # (B, 7)

        # Concatenate: (B, T*4 + 7)
        node_features = torch.cat([temporal_flat, maternal], dim=1)
        node_features = node_features.unsqueeze(-1)  # (B, N, 1)

        # Node mask: temporal nodes are valid only where scan exists
        temporal_mask = batch['temporal_mask']  # (B, T)
        # Expand to per-variable: (B, T) → (B, T*4)
        temporal_node_mask = temporal_mask.unsqueeze(-1).expand(
            B, T, N_US_TYPES
        ).reshape(B, T * N_US_TYPES)
        # Maternal nodes are always valid
        maternal_mask = torch.ones(B, self.n_maternal, device=temporal_mask.device)
        node_mask = torch.cat([temporal_node_mask, maternal_mask], dim=1)

        return {
            'node_features': node_features,
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'node_types': self.node_types,
            'node_mask': node_mask,
        }
