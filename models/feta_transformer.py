"""
FETA-Transformer — Fetal Echocardiographic Temporal Attention Transformer
=========================================================================

A Transformer-based model for predicting first-trimester pregnancy loss
from longitudinal ultrasound measurements.

Key innovations over standard Transformers:
  1. Continuous positional encoding using actual gestational age (weeks)
  2. Modality-specific input projections (FHR, CRL, GS, YSD)
  3. Attention-based temporal pooling (learnable query)
  4. Maternal cross-attention (static features attend to temporal repr.)

References:
  - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
  - Doubilet & Benson, 1995 (FHR curves)
  - Hadlock, 1992 (CRL curves)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Continuous Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class ContinuousPositionalEncoding(nn.Module):
    """
    Positional encoding using actual gestational age instead of sequence index.

    PE(t, 2i)   = sin(t / θ^(2i/d_model))
    PE(t, 2i+1) = cos(t / θ^(2i/d_model))

    where t is gestational age in weeks, θ=10000.
    This captures that the biological gap between week 6→7 differs from 9→10.
    """

    def __init__(self, d_model: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.theta = theta

        # Pre-compute division terms: θ^(2i/d_model) for i=0..d_model/2-1
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(theta) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, gestational_ages: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        gestational_ages : (B, T) — actual GA in weeks

        Returns
        -------
        pe : (B, T, d_model)
        """
        B, T = gestational_ages.shape
        # ga: (B, T, 1), div_term: (d_model/2,) -> (1, 1, d_model/2)
        ga = gestational_ages.unsqueeze(-1)  # (B, T, 1)
        div = self.div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, d/2)

        # div stores theta^(-2i/d), so multiplication is equivalent to
        # t / theta^(2i/d).
        angles = ga * div  # (B, T, d/2)

        pe = torch.zeros(B, T, self.d_model, device=gestational_ages.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe


# ──────────────────────────────────────────────────────────────────────────────
# Modality-Specific Input Projection
# ──────────────────────────────────────────────────────────────────────────────

class ModalityProjection(nn.Module):
    """
    Separate linear projections for each ultrasound measurement type.
    Each modality (FHR, CRL, GS, YSD) gets its own W_m, b_m → d_model.
    The projections are summed to create the input embedding.
    """

    def __init__(self, n_modalities: int = 4, d_model: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_modalities)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, n_modalities) — raw measurement values

        Returns
        -------
        h : (B, T, d_model) — sum of modality-specific projections
        """
        h = torch.zeros(
            x.shape[0], x.shape[1], self.projections[0].out_features,
            device=x.device, dtype=x.dtype
        )
        for i, proj in enumerate(self.projections):
            h = h + proj(x[:, :, i:i+1])  # (B, T, 1) → (B, T, d_model)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# Attention-Based Temporal Pooling
# ──────────────────────────────────────────────────────────────────────────────

class MaternalTokenProjection(nn.Module):
    """
    Separate linear projections for each maternal variable → d_model.

    Each of the 7 maternal variables (age, bmi, parity, gravidity, prev_loss,
    conception, singleton) becomes its own token, mirroring how PREG-Net treats
    them as 7 peer graph nodes. This is what makes the FETA/PREG-Net comparison
    apples-to-apples: both give every maternal variable its own representation.
    """

    def __init__(self, n_maternal: int = 7, d_model: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_maternal)
        ])

    def forward(self, maternal: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        maternal : (B, n_maternal) — static maternal features

        Returns
        -------
        m : (B, n_maternal, d_model) — one token per maternal variable
        """
        tokens = [
            proj(maternal[:, i:i+1])            # (B, 1) → (B, d_model)
            for i, proj in enumerate(self.projections)
        ]
        return torch.stack(tokens, dim=1)       # (B, n_maternal, d_model)


class AttentionPooling(nn.Module):
    """
    Learnable query vector that computes attention weights over time steps:
        α_t = softmax(w^T h_t)
        z = Σ α_t · h_t

    This replaces [CLS] token and lets us extract which gestational
    weeks are most predictive of outcome.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h    : (B, T, d_model) — encoder outputs
        mask : (B, T) — 1.0 for valid, 0.0 for padding

        Returns
        -------
        z       : (B, d_model)
        weights : (B, T) — attention weights (for explainability)
        """
        # Project and compute scores
        h_proj = torch.tanh(self.proj(h))  # (B, T, d_model)
        scores = torch.einsum('btd,d->bt', h_proj, self.query)  # (B, T)

        # Mask padded positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)  # (B, T)
        z = torch.einsum('bt,btd->bd', weights, h)  # (B, d_model)

        return z, weights


# ──────────────────────────────────────────────────────────────────────────────
# Maternal Cross-Attention
# ──────────────────────────────────────────────────────────────────────────────

class MaternalCrossAttention(nn.Module):
    """
    Static maternal features attend over the *token sequence*.

    Q = W_q · maternal    (projected maternal features as queries)
    K = W_k · h           (per-time-step tokens as keys)
    V = W_v · h           (per-time-step tokens as values)

    Lets the model weight different points of the trajectory according to the
    patient's maternal profile, and exposes which timepoints a given profile
    makes salient.

    Keys/values MUST be the full ``(B, T, d)`` token sequence, never a pooled
    ``(B, 1, d)`` vector: a softmax over a single key is identically 1.0, which
    drops Q — and with it the maternal features — out of the forward pass
    entirely. That was the case here until 2026-07-16; the module returned
    plausible-looking *pre-softmax* scores while contributing exactly zero
    gradient to the maternal inputs.
    """

    def __init__(
        self,
        d_model: int,
        n_maternal: int = 7,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(n_maternal, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        temporal_repr: torch.Tensor,
        maternal: torch.Tensor,
        token_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        temporal_repr : (B, d_model) — pooled temporal representation (residual base)
        maternal      : (B, n_maternal) — static maternal features
        token_seq     : (B, T, d_model) — per-time-step tokens (keys/values)
        token_mask    : (B, T) — 1.0 for real scans, 0.0 for padding

        Returns
        -------
        out    : (B, d_model)
        attn_w : (B, n_heads, T) — post-softmax weights over timepoints
        """
        B, T, _ = token_seq.shape
        H, dh = self.n_heads, self.d_head

        Q = self.W_q(maternal).view(B, 1, H, dh).transpose(1, 2)   # (B, H, 1, dh)
        K = self.W_k(token_seq).view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)
        V = self.W_v(token_seq).view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)  # (B, H, 1, T)

        if token_mask is not None:
            ignore = (token_mask == 0).view(B, 1, 1, T)  # True = IGNORE
            scores = scores.masked_fill(ignore, -1e9)

        attn_w = F.softmax(scores, dim=-1)  # (B, H, 1, T) — over T keys, so Q survives
        context = torch.matmul(self.dropout(attn_w), V)              # (B, H, 1, dh)
        context = context.transpose(1, 2).contiguous().view(B, -1)   # (B, d_model)

        out = self.layer_norm(self.W_o(context) + temporal_repr)     # residual

        return out, attn_w.squeeze(2)  # (B, n_heads, T)


# ──────────────────────────────────────────────────────────────────────────────
# FETA-Transformer
# ──────────────────────────────────────────────────────────────────────────────

class FETATransformer(nn.Module):
    """
    Fetal Echocardiographic Temporal Attention Transformer.

    End-to-end architecture:
        1. Modality-specific projections → d_model embeddings
        2. Continuous positional encoding (gestational age)
        3. Transformer encoder (self-attention over time steps)
        4. Attention-based temporal pooling
        5. Maternal cross-attention
        6. Classification head → logits

    Parameters
    ----------
    d_model : int
        Model dimension (default: 64).
    n_heads : int
        Number of attention heads (default: 4).
    n_layers : int
        Number of Transformer encoder layers (default: 2).
    d_ff : int
        Feed-forward dimension (default: 128).
    dropout : float
        Dropout rate (default: 0.2).
    n_us_features : int
        Number of ultrasound features (default: 4).
    n_maternal : int
        Number of maternal features (default: 7).
    inject_maternal_tokens : bool
        If ``True`` (default), each maternal variable enters the encoder as its own
        token, so scan tokens and maternal tokens co-attend in every layer — the
        same peer treatment PREG-Net's graph gives them. If ``False``, maternal
        reaches the model only through :class:`MaternalCrossAttention` at the head.
        Set ``False`` only to reproduce the pre-2026-07-16 ablation.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.2,
        n_us_features: int = 4,
        n_maternal: int = 7,
        inject_maternal_tokens: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_maternal = n_maternal
        self.inject_maternal_tokens = inject_maternal_tokens

        # 1. Modality-specific input projections
        self.modality_proj = ModalityProjection(n_us_features, d_model)

        # 1b. One token per maternal variable (peers of the scan tokens)
        self.maternal_proj = MaternalTokenProjection(n_maternal, d_model)
        self.maternal_norm = nn.LayerNorm(d_model)

        # 2. Continuous positional encoding
        self.pos_encoder = ContinuousPositionalEncoding(d_model)

        # 3. Input LayerNorm + Dropout
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 5. Attention-based temporal pooling
        self.attn_pool = AttentionPooling(d_model)

        # 6. Maternal cross-attention
        self.maternal_cross_attn = MaternalCrossAttention(
            d_model=d_model,
            n_maternal=n_maternal,
            n_heads=min(2, n_heads),
            dropout=dropout,
        )

        # 7. Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_tokens(
        self,
        temporal_features: torch.Tensor,
        gestational_ages: torch.Tensor,
        temporal_mask: torch.Tensor,
        maternal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the encoder up to (but not including) temporal pooling.

        This is the fusion point for intermediate cross-attention: it returns the
        per-time-step token sequence before it is collapsed by :meth:`head`.

        When ``inject_maternal_tokens`` is set and ``maternal_features`` is given,
        the 7 maternal tokens are prepended for the encoder pass and dropped from
        the return value. Self-attention has by then written maternal context into
        every scan token, so the maternal signal survives while the returned shape
        stays ``(B, T, d)`` — which the hybrid's cross-modal block and
        ``temporal_mask`` both depend on.

        Parameters
        ----------
        temporal_features : (B, T, 4) — FHR, CRL, GS, YSD
        gestational_ages  : (B, T) — actual gestational age in weeks
        temporal_mask     : (B, T) — 1.0 for real scans, 0.0 for padding
        maternal_features : (B, n_maternal), optional — static maternal features

        Returns
        -------
        h : (B, T, d_model) — contextualized token sequence
        """
        # 1. Modality-specific projections
        h = self.modality_proj(temporal_features)  # (B, T, d_model)

        # 2. Add continuous positional encoding
        pe = self.pos_encoder(gestational_ages)    # (B, T, d_model)
        h = h + pe

        # 3. Input normalization
        h = self.input_norm(h)
        h = self.input_dropout(h)

        # 4. Transformer encoder
        # Create attention mask: True = IGNORE, so invert mask
        src_key_padding_mask = (temporal_mask == 0)  # (B, T)

        if self.inject_maternal_tokens and maternal_features is not None:
            T = h.shape[1]
            # Maternal tokens carry no positional encoding — they are not events in time.
            m = self.maternal_norm(self.maternal_proj(maternal_features))  # (B, M, d)
            h = torch.cat([m, h], dim=1)                                   # (B, M+T, d)
            m_pad = torch.zeros(
                h.shape[0], self.n_maternal,
                dtype=src_key_padding_mask.dtype, device=h.device,
            )  # maternal tokens are always valid keys
            src_key_padding_mask = torch.cat([m_pad, src_key_padding_mask], dim=1)

            h = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
            h = h[:, self.n_maternal:, :]  # (B, T, d) — scan tokens, maternal absorbed
        else:
            h = self.transformer_encoder(
                h, src_key_padding_mask=src_key_padding_mask
            )  # (B, T, d_model)

        return h

    def head(
        self,
        h: torch.Tensor,
        temporal_mask: torch.Tensor,
        maternal_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Pool the token sequence, apply maternal cross-attention, and classify.

        Parameters
        ----------
        h                 : (B, T, d_model) — token sequence from :meth:`encode_tokens`
        temporal_mask     : (B, T) — 1.0 for real scans, 0.0 for padding
        maternal_features : (B, 7) — static maternal features

        Returns
        -------
        logits : (B, 1) — raw logits
        attn   : dict with 'temporal_pooling_weights' (B, T) and
                 'cross_attention_weights' (B, n_heads, T)
        z      : (B, d_model) — penultimate pooled representation
        """
        # 5. Attention-based temporal pooling
        z, pool_weights = self.attn_pool(h, mask=temporal_mask)  # (B, d_model)

        # 6. Maternal cross-attention (over the token sequence, not the pooled vector)
        z, cross_weights = self.maternal_cross_attn(
            z, maternal_features, h, temporal_mask
        )

        # 7. Classification
        logits = self.classifier(z)  # (B, 1)

        attention_weights = {
            'temporal_pooling_weights': pool_weights,
            'cross_attention_weights': cross_weights,
        }

        return logits, attention_weights, z

    def forward(
        self,
        temporal_features: torch.Tensor,
        gestational_ages: torch.Tensor,
        temporal_mask: torch.Tensor,
        maternal_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        temporal_features : (B, T, 4) — FHR, CRL, GS, YSD
        gestational_ages  : (B, T) — actual gestational age in weeks
        temporal_mask     : (B, T) — 1.0 for real scans, 0.0 for padding
        maternal_features : (B, 7) — static maternal features

        Returns
        -------
        logits : (B, 1) — raw logits (apply sigmoid for probabilities)
        attn   : dict with:
            'temporal_pooling_weights' : (B, T)
            'cross_attention_weights'  : (B, n_heads, T)
        """
        h = self.encode_tokens(
            temporal_features, gestational_ages, temporal_mask, maternal_features
        )
        logits, attention_weights, _ = self.head(h, temporal_mask, maternal_features)
        return logits, attention_weights
