"""
PREG-Net — Pregnancy Risk Evaluation with Graph-based Multimodal Fusion Network
=================================================================================

A Graph Attention Network (GAT) that explicitly models physiological relationships
between clinical variables for pregnancy loss prediction.

Key innovations:
  1. Knowledge-guided graph structure (not learned from data)
  2. Dynamic edge weights via attention (patient-specific reasoning)
  3. Temporal graph extension (same variable across time steps)
  4. Node-level explainability (which variables drove the prediction)

All implemented in pure PyTorch — no PyTorch Geometric dependency.

References:
  - Veličković et al., "Graph Attention Networks", ICLR 2018
  - Research PDF: PREG-Net architecture specification
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Custom GAT Layer (pure PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Graph Attention Network layer implemented in pure PyTorch.

    Computes attention-weighted message passing:
        α_vu = softmax_u(LeakyReLU(a^T [Wh_v || Wh_u]))
        h'_v = σ(Σ_u α_vu · W h_u)

    For small graphs (< 50 nodes), uses dense operations for efficiency.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension per head.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    leaky_relu_slope : float
        Negative slope for LeakyReLU.
    concat : bool
        If True, concatenate heads; if False, average them.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        leaky_relu_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat

        # Linear transformation per head
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        # Attention parameters per head
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim, 1))

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(n_heads * out_dim if concat else out_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x          : (B, N, in_dim) — node features
        edge_index : (2, E) — COO edge list (shared across batch)
        node_mask  : (B, N) — 1.0 for valid nodes, 0.0 for padding

        Returns
        -------
        out        : (B, N, n_heads*out_dim) if concat else (B, N, out_dim)
        edge_attn  : (B, n_heads, E) — attention weights per edge
        """
        B, N, _ = x.shape
        H = self.n_heads
        src, dst = edge_index[0], edge_index[1]  # (E,)
        E = src.shape[0]

        # Linear transform: (B, N, in) @ (H, in, out) → (B, H, N, out)
        Wh = torch.einsum('bni,hio->bhno', x, self.W)  # (B, H, N, out)

        # Compute attention scores
        # Source scores: (B, H, N, out) @ (H, out, 1) → (B, H, N, 1)
        score_src = torch.einsum('bhno,hoi->bhni', Wh, self.a_src).squeeze(-1)  # (B,H,N)
        score_dst = torch.einsum('bhno,hoi->bhni', Wh, self.a_dst).squeeze(-1)  # (B,H,N)

        # Gather scores for edge endpoints
        # score_src[src] + score_dst[dst]
        e_src = score_src[:, :, src]  # (B, H, E)
        e_dst = score_dst[:, :, dst]  # (B, H, E)

        edge_scores = self.leaky_relu(e_src + e_dst)  # (B, H, E)

        # Mask invalid source nodes
        if node_mask is not None:
            src_valid = node_mask[:, src]  # (B, E)
            edge_scores = edge_scores.masked_fill(
                src_valid.unsqueeze(1) == 0, -1e9
            )

        # Softmax per destination node (sparse-friendly via scatter)
        edge_attn = self._sparse_softmax(edge_scores, dst, N)  # (B, H, E)
        edge_attn = self.dropout(edge_attn)

        # Message passing: aggregate neighbor features weighted by attention
        # Wh[:, :, src, :] → (B, H, E, out), weight by edge_attn → scatter to dst
        messages = Wh[:, :, src, :] * edge_attn.unsqueeze(-1)  # (B, H, E, out)

        # Scatter add to destination nodes
        out = torch.zeros(B, H, N, self.out_dim, device=x.device, dtype=x.dtype)
        dst_expanded = dst.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        dst_expanded = dst_expanded.expand(B, H, E, self.out_dim)
        out.scatter_add_(2, dst_expanded, messages)

        # Combine heads
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * self.out_dim)
        else:
            out = out.mean(dim=1)  # (B, N, out_dim)

        out = out + self.bias

        return out, edge_attn

    @staticmethod
    def _sparse_softmax(
        scores: torch.Tensor,
        indices: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over scores grouped by destination node index.

        Parameters
        ----------
        scores  : (B, H, E)
        indices : (E,) — destination node indices
        n_nodes : int

        Returns
        -------
        attn : (B, H, E) — normalized attention weights
        """
        B, H, E = scores.shape

        # Compute max per destination for numerical stability
        idx = indices.unsqueeze(0).unsqueeze(0).expand(B, H, E)
        max_vals = torch.full((B, H, n_nodes), -1e9, device=scores.device)
        max_vals.scatter_reduce_(2, idx, scores, reduce='amax')
        scores_shifted = scores - max_vals.gather(2, idx)

        exp_scores = torch.exp(scores_shifted)

        # Sum per destination
        sum_exp = torch.zeros(B, H, n_nodes, device=scores.device)
        sum_exp.scatter_add_(2, idx, exp_scores)

        # Normalize
        attn = exp_scores / (sum_exp.gather(2, idx) + 1e-10)

        return attn


# ──────────────────────────────────────────────────────────────────────────────
# Node-Level Readout
# ──────────────────────────────────────────────────────────────────────────────

class GraphReadout(nn.Module):
    """
    Attention-weighted aggregation of all node embeddings:
        β_v = softmax(MLP(h_v))
        z = Σ β_v · h_v
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h         : (B, N, hidden_dim)
        node_mask : (B, N) — 1.0 for valid, 0.0 for padding

        Returns
        -------
        z         : (B, hidden_dim)
        node_imp  : (B, N) — node importance scores
        """
        scores = self.gate(h).squeeze(-1)  # (B, N)

        if node_mask is not None:
            scores = scores.masked_fill(node_mask == 0, -1e9)

        node_imp = F.softmax(scores, dim=-1)  # (B, N)
        z = torch.einsum('bn,bnd->bd', node_imp, h)  # (B, hidden_dim)

        return z, node_imp


# ──────────────────────────────────────────────────────────────────────────────
# PREG-Net
# ──────────────────────────────────────────────────────────────────────────────

class PREGNet(nn.Module):
    """
    Pregnancy Risk Evaluation with Graph-based Multimodal Fusion Network.

    Architecture:
        1. Node feature encoding (type embedding + value projection)
        2. Stacked GAT layers with residual connections
        3. Attention-weighted graph readout
        4. Classification head → logits

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for node embeddings (default: 64).
    n_gat_layers : int
        Number of GAT layers (default: 2).
    n_heads : int
        Number of attention heads per GAT layer (default: 4).
    dropout : float
        Dropout rate (default: 0.2).
    n_node_types : int
        Number of distinct node types (default: 11).
    leaky_relu_slope : float
        LeakyReLU slope for GAT attention (default: 0.2).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_gat_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        n_node_types: int = 11,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Node feature encoding
        self.type_embedding = nn.Embedding(n_node_types, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)

        # 2. Stacked GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for i in range(n_gat_layers):
            in_dim = hidden_dim
            # For intermediate layers with concat, output is n_heads * out_dim
            out_per_head = hidden_dim // n_heads
            is_last = (i == n_gat_layers - 1)

            self.gat_layers.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=out_per_head,
                    n_heads=n_heads,
                    dropout=dropout,
                    leaky_relu_slope=leaky_relu_slope,
                    concat=True,  # concat heads → out_dim = n_heads * out_per_head = hidden_dim
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.gat_dropout = nn.Dropout(dropout)

        # 3. Graph readout
        self.readout = GraphReadout(hidden_dim)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        node_features : (B, N, 1) — scalar feature value per node
        edge_index    : (2, E) — graph connectivity (shared across batch)
        node_types    : (N,) — integer type for each node
        node_mask     : (B, N) — 1.0 for valid, 0.0 for padded

        Returns
        -------
        logits : (B, 1) — raw logits
        attn   : dict with:
            'node_importance' : (B, N)
            'edge_attention'  : list of (B, n_heads, E) per layer
        """
        B = node_features.shape[0]

        # 1. Node feature encoding
        type_emb = self.type_embedding(node_types)  # (N, hidden) or (B, N, hidden)
        if type_emb.dim() == 2:
            type_emb = type_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden)
        val_emb = self.value_proj(node_features)  # (B, N, hidden)
        h = type_emb + val_emb

        # Mask invalid nodes
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)

        # 2. GAT layers with residual connections
        edge_attentions = []
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            residual = h
            h, edge_attn = gat(h, edge_index, node_mask)
            h = F.elu(h)
            h = self.gat_dropout(h)
            h = norm(h + residual)  # residual connection

            if node_mask is not None:
                h = h * node_mask.unsqueeze(-1)

            edge_attentions.append(edge_attn)

        # 3. Graph readout
        z, node_imp = self.readout(h, node_mask)  # (B, hidden)

        # 4. Classification
        logits = self.classifier(z)  # (B, 1)

        attention_weights = {
            'node_importance': node_imp,
            'edge_attention': edge_attentions,
        }

        return logits, attention_weights
