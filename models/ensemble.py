"""
Ensemble model combining FETA-Transformer and PREG-Net.

Supports two fusion strategies:
  - **Average fusion** (default): Averages the raw logits from both models.
  - **Learned fusion**: A small linear layer learns optimal weights for
    combining the two model outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .feta_transformer import FETATransformer
from .preg_net import PREGNet


class EnsemblePredictor(nn.Module):
    """Late-fusion ensemble of FETA-Transformer and PREG-Net.

    Combines pregnancy loss predictions from a temporal Transformer model
    and a graph-based model.  Returns the fused logits, individual model
    logits, and all attention weights from both sub-models for
    interpretability.

    Args:
        feta_kwargs: Keyword arguments forwarded to :class:`FETATransformer`.
        preg_kwargs: Keyword arguments forwarded to :class:`PREGNet`.
        learned_fusion: If ``True``, use a learned 2→1 linear layer for
            combining logits instead of simple averaging.

    Example::

        model = EnsemblePredictor(learned_fusion=True)
        out = model(temporal_features, gestational_ages, temporal_mask,
                     maternal_features, node_features, edge_index, node_types)
        logits = out["logits"]
    """

    def __init__(
        self,
        feta_kwargs: Optional[Dict[str, Any]] = None,
        preg_kwargs: Optional[Dict[str, Any]] = None,
        learned_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.feta = FETATransformer(**(feta_kwargs or {}))
        self.preg = PREGNet(**(preg_kwargs or {}))

        self.learned_fusion = learned_fusion
        if learned_fusion:
            # 2 input logits -> 1 combined logit
            self.fusion_layer = nn.Linear(2, 1, bias=True)
            # Initialize to equal weighting
            nn.init.constant_(self.fusion_layer.weight, 0.5)
            nn.init.constant_(self.fusion_layer.bias, 0.0)

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
        """Forward pass through both models and fuse.

        Args:
            temporal_features: (B, T, 4) — FHR, CRL, GS, YSD.
            gestational_ages: (B, T) — gestational age in weeks.
            temporal_mask: (B, T) — 1.0 real, 0.0 padding.
            maternal_features: (B, 7) — static maternal features.
            node_features: (B, N, 1) — graph node scalar values.
            edge_index: (2, E) — graph connectivity (COO, shared across batch).
            node_types: (N,) — integer node type ids.
            node_mask: (B, N) — 1.0 valid, 0.0 padded.

        Returns:
            Dictionary with:
                - ``logits``: (B, 1) combined logits (no sigmoid).
                - ``feta_logits``: (B, 1) FETA-Transformer raw logits.
                - ``preg_logits``: (B, 1) PREG-Net raw logits.
                - ``feta_attention``: dict of FETA attention weights.
                - ``preg_attention``: dict of PREG-Net attention weights.
                - ``fusion_weights``: (2,) effective fusion weights
                  (learned or [0.5, 0.5]).
        """
        # Run both sub-models
        feta_logits, feta_attn = self.feta(
            temporal_features, gestational_ages, temporal_mask, maternal_features
        )
        preg_logits, preg_attn = self.preg(
            node_features, edge_index, node_types, node_mask
        )

        # Fuse logits
        if self.learned_fusion:
            stacked = torch.cat([feta_logits, preg_logits], dim=-1)  # (B, 2)
            combined_logits = self.fusion_layer(stacked)             # (B, 1)
            fusion_weights = self.fusion_layer.weight.detach().squeeze()
        else:
            combined_logits = (feta_logits + preg_logits) / 2.0
            fusion_weights = torch.tensor(
                [0.5, 0.5], device=feta_logits.device, dtype=feta_logits.dtype
            )

        return {
            "logits": combined_logits,
            "feta_logits": feta_logits,
            "preg_logits": preg_logits,
            "feta_attention": feta_attn,
            "preg_attention": preg_attn,
            "fusion_weights": fusion_weights,
        }


# Backward-compatible name used by older imports.
EnsembleModel = EnsemblePredictor
