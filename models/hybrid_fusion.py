"""
Hybrid Cross-Attention Fusion of FETA-Transformer and PREG-Net
===============================================================

An **intermediate-fusion** model in which the temporal Transformer and the graph
attention network exchange information *mid-network* before either commits to a
prediction, rather than the late-fusion :class:`~models.ensemble.EnsemblePredictor`
which only averages/combines the two final logits.

Flow
----
1. FETA encodes its per-time-step tokens ``h_feta (B, T, d)`` (before pooling).
2. PREG-Net encodes its per-node embeddings ``h_preg (B, N, d)`` (before readout).
3. :class:`CrossModalFusion` lets the two representations attend to each other
   (bidirectional, masked), updating both in place.
4. Each model finishes its own head → pooled embeddings ``z_feta``/``z_preg`` and
   auxiliary per-branch logits.
5. A fusion head over ``[z_feta ⊕ z_preg ⊕ feta_logit ⊕ preg_logit]`` produces the
   single combined logit.

The auxiliary per-branch logits are returned so the trainer can add a per-branch
loss that keeps both towers individually predictive (prevents the branch collapse
observed in the averaged ensemble).

Pure PyTorch — no PyTorch Geometric, consistent with the rest of the project.

References
----------
- Frontiers in AI (2025), "Multimodal graph neural networks in healthcare: a review
  of fusion strategies" — intermediate/attention fusion is the dominant, top-performing
  strategy for combining sequence and graph encoders in clinical prediction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feta_transformer import FETATransformer
from .preg_net import PREGNet


# ──────────────────────────────────────────────────────────────────────────────
# Masked multi-head cross-attention (one direction)
# ──────────────────────────────────────────────────────────────────────────────

class _CrossAttention(nn.Module):
    """Multi-head cross-attention with residual + LayerNorm.

    Queries come from ``x``; keys/values from ``y``. Padded key positions are
    masked out so a query never attends to padding.

    Parameters
    ----------
    d_model : int
        Model dimension (must match both towers).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to the attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x        : (B, Lq, d_model) — query sequence (updated and returned)
        y        : (B, Lk, d_model) — key/value sequence (the "other" tower)
        key_mask : (B, Lk) — 1.0 for valid key positions, 0.0 for padding

        Returns
        -------
        out  : (B, Lq, d_model) — x updated with cross-modal context (residual + LN)
        attn : (B, n_heads, Lq, Lk) — attention weights (for explainability)
        """
        B, Lq, D = x.shape
        Lk = y.shape[1]
        H, dh = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, Lq, H, dh).transpose(1, 2)  # (B, H, Lq, dh)
        K = self.W_k(y).view(B, Lk, H, dh).transpose(1, 2)  # (B, H, Lk, dh)
        V = self.W_v(y).view(B, Lk, H, dh).transpose(1, 2)  # (B, H, Lk, dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)  # (B, H, Lq, Lk)

        if key_mask is not None:
            # True = IGNORE. (B, Lk) -> (B, 1, 1, Lk)
            ignore = (key_mask == 0).view(B, 1, 1, Lk)
            scores = scores.masked_fill(ignore, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B, H, Lq, Lk)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, H, Lq, dh)
        context = context.transpose(1, 2).contiguous().view(B, Lq, D)  # (B, Lq, d_model)

        out = self.norm(x + self.W_o(context))  # residual + LayerNorm
        return out, attn


# ──────────────────────────────────────────────────────────────────────────────
# Bidirectional cross-modal fusion block
# ──────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """Let FETA tokens and PREG-Net nodes attend to each other.

    Parameters
    ----------
    d_model : int
        Shared model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout.
    bidirectional : bool
        If ``True`` (default), both directions run. If ``False``, only FETA attends
        to PREG (a cheaper, unidirectional ablation) and PREG is passed through.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.feta_to_preg = _CrossAttention(d_model, n_heads, dropout)
        if bidirectional:
            self.preg_to_feta = _CrossAttention(d_model, n_heads, dropout)

    def forward(
        self,
        h_feta: torch.Tensor,
        h_preg: torch.Tensor,
        temporal_mask: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Parameters
        ----------
        h_feta        : (B, T, d_model) — FETA token sequence (queries nodes)
        h_preg        : (B, N, d_model) — PREG-Net node embeddings (queries tokens)
        temporal_mask : (B, T) — valid scans (masks tokens as keys)
        node_mask     : (B, N) — valid nodes (masks nodes as keys)

        Returns
        -------
        h_feta' : (B, T, d_model) — FETA tokens updated with graph context
        h_preg' : (B, N, d_model) — PREG nodes updated with temporal context
        attn    : {'feta_to_preg': (B, H, T, N), 'preg_to_feta': (B, H, N, T) or None}
        """
        new_feta, f2p = self.feta_to_preg(h_feta, h_preg, key_mask=node_mask)
        if self.bidirectional:
            new_preg, p2f = self.preg_to_feta(h_preg, h_feta, key_mask=temporal_mask)
        else:
            new_preg, p2f = h_preg, None
        return new_feta, new_preg, {'feta_to_preg': f2p, 'preg_to_feta': p2f}


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid predictor
# ──────────────────────────────────────────────────────────────────────────────

class HybridFusionPredictor(nn.Module):
    """Intermediate cross-attention fusion of FETA-Transformer and PREG-Net.

    Drop-in replacement for :class:`~models.ensemble.EnsemblePredictor`: it takes the
    same forward arguments and returns the same output-dict keys (plus a new
    ``cross_modal_attention`` entry), so it slots into the ``collate_mode="ensemble"``
    training path and the ensemble evaluation script unchanged.

    Parameters
    ----------
    feta_kwargs : dict, optional
        Forwarded to :class:`FETATransformer`.
    preg_kwargs : dict, optional
        Forwarded to :class:`PREGNet`.
    n_heads : int
        Heads for the cross-modal attention.
    dropout : float
        Dropout for the fusion block and head.
    bidirectional : bool
        Whether the two towers attend to each other in both directions.
    """

    def __init__(
        self,
        feta_kwargs: Optional[Dict[str, Any]] = None,
        preg_kwargs: Optional[Dict[str, Any]] = None,
        n_heads: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.feta = FETATransformer(**(feta_kwargs or {}))
        self.preg = PREGNet(**(preg_kwargs or {}))
        self.gated = gated

        d_feta = self.feta.d_model
        d_preg = self.preg.hidden_dim
        if d_feta != d_preg:
            raise ValueError(
                f"Cross-attention requires matching dims, got FETA d_model={d_feta} "
                f"and PREG hidden_dim={d_preg}. Set feta_kwargs/preg_kwargs to align them."
            )
        d = d_feta

        self.fusion = CrossModalFusion(
            d_model=d, n_heads=n_heads, dropout=dropout, bidirectional=bidirectional
        )

        if gated:
            # Gated pure-branch fusion. The per-branch (aux) heads run on the PURE
            # pre-fusion representations so the two towers stay complementary (they do
            # not collapse into copies of each other, as bidirectional cross-attention
            # otherwise forces). The cross-attended representations instead drive a
            # learned per-patient gate that ROUTES between the two branch predictions,
            # plus a small correction term. This is what lets fusion exploit the
            # patients where exactly one tower is right.
            self.fusion_norm = nn.LayerNorm(2 * d)
            self.gate_head = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1)
            )
            self.correction_head = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1)
            )
        else:
            # Original concat fusion: fuse the two pooled embeddings; the two branch
            # logits are appended so the head generalizes a learned-logit combiner.
            self.fusion_norm = nn.LayerNorm(2 * d)
            self.fusion_head = nn.Sequential(
                nn.Linear(2 * d + 2, d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, 1),
            )
        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        heads = ([self.gate_head, self.correction_head] if self.gated
                 else [self.fusion_head])
        for head in heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        # -- FETA-Transformer inputs --
        temporal_features: torch.Tensor,
        gestational_ages: torch.Tensor,
        temporal_mask: torch.Tensor,
        maternal_features: torch.Tensor,
        # -- PREG-Net inputs --
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with mid-network cross-attention fusion.

        Returns a dict with:
            - ``logits``: (B, 1) fused logits (no sigmoid).
            - ``feta_logits`` / ``preg_logits``: (B, 1) auxiliary per-branch logits.
            - ``feta_attention`` / ``preg_attention``: each tower's attention dicts.
            - ``cross_modal_attention``: {'feta_to_preg', 'preg_to_feta'} weights.
            - ``fusion_weights``: ``None`` (fusion is a learned head, not a 2-vector).
        """
        # 1. Encode each tower up to its fusion point (pure, pre-fusion).
        #    maternal_features must reach encode_tokens: without it the FETA tower is
        #    maternal-blind and the hybrid sees clinical variables only through PREG-Net.
        h_feta_pure = self.feta.encode_tokens(
            temporal_features, gestational_ages, temporal_mask, maternal_features
        )
        h_preg_pure, edge_attentions = self.preg.encode_nodes(
            node_features, edge_index, node_types, node_mask
        )

        # 2. Let the two representations attend to each other.
        h_feta_x, h_preg_x, cross_attn = self.fusion(
            h_feta_pure, h_preg_pure, temporal_mask, node_mask
        )

        fusion_weights = None
        if self.gated:
            # Branch (aux) heads run on the PURE reps -> towers stay complementary.
            feta_logits, feta_attn, _ = self.feta.head(
                h_feta_pure, temporal_mask, maternal_features
            )
            preg_logits, preg_attn, _ = self.preg.head(h_preg_pure, node_mask)
            preg_attn['edge_attention'] = edge_attentions

            # Cross-attended pooled reps drive a per-patient gate + correction.
            _, _, zx_feta = self.feta.head(h_feta_x, temporal_mask, maternal_features)
            _, _, zx_preg = self.preg.head(h_preg_x, node_mask)
            zx = self.fusion_norm(torch.cat([zx_feta, zx_preg], dim=-1))  # (B, 2d)
            gate = torch.sigmoid(self.gate_head(zx))                      # (B, 1)
            correction = self.correction_head(zx)                        # (B, 1)
            logits = gate * feta_logits + (1.0 - gate) * preg_logits + correction
            fusion_weights = gate  # per-patient FETA weight (explainability)
        else:
            # Original: branch heads + concat fusion on the fused representations.
            feta_logits, feta_attn, z_feta = self.feta.head(
                h_feta_x, temporal_mask, maternal_features
            )
            preg_logits, preg_attn, z_preg = self.preg.head(h_preg_x, node_mask)
            preg_attn['edge_attention'] = edge_attentions
            z = self.fusion_norm(torch.cat([z_feta, z_preg], dim=-1))      # (B, 2d)
            fused_in = torch.cat([z, feta_logits, preg_logits], dim=-1)    # (B, 2d + 2)
            logits = self.fusion_head(fused_in)                           # (B, 1)

        return {
            "logits": logits,
            "feta_logits": feta_logits,
            "preg_logits": preg_logits,
            "feta_attention": feta_attn,
            "preg_attention": preg_attn,
            "cross_modal_attention": cross_attn,
            "fusion_weights": fusion_weights,
        }
