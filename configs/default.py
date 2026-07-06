"""
Configuration dataclasses for FETA-Transformer and PREG-Net.
All hyperparameters are centralized here for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    max_scans: int = 5
    us_features: List[str] = field(default_factory=lambda: ['FHR', 'CRL', 'GS', 'YSD'])
    maternal_features: List[str] = field(
        default_factory=lambda: ['age', 'bmi', 'parity', 'gravidity',
                                 'previous_loss', 'conception_ivf', 'singleton']
    )
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class FETATransformerConfig:
    """FETA-Transformer architecture configuration."""
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2
    n_us_features: int = 4       # FHR, CRL, GS, YSD
    n_maternal_features: int = 7  # age, bmi, parity, gravidity, prev_loss, ivf, singleton
    max_seq_len: int = 5
    pe_theta: float = 10000.0


@dataclass
class PREGNetConfig:
    """PREG-Net architecture configuration."""
    hidden_dim: int = 64
    n_gat_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    n_node_types: int = 11  # 4 US vars + 7 maternal vars
    max_scans: int = 5
    leaky_relu_slope: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    patience: int = 15
    seed: int = 42
    pos_weight: float = 3.5  # ~78/22 class ratio
    scheduler_T_0: int = 20  # cosine annealing restart period
    scheduler_T_mult: int = 2
    grad_clip_norm: float = 1.0


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_folds: int = 5
    bootstrap_iterations: int = 1000
    ci_level: float = 0.95
    threshold: float = 0.5


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    feta: FETATransformerConfig = field(default_factory=FETATransformerConfig)
    preg: PREGNetConfig = field(default_factory=PREGNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_default_config() -> Config:
    """Return the default configuration."""
    return Config()
