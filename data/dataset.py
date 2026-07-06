"""
PyTorch Dataset and DataLoader utilities for pregnancy loss prediction.

Handles:
  - Loading patients.csv and scans.csv
  - Padding/truncating longitudinal scans to fixed length T=5
  - Building temporal feature tensors for FETA-Transformer
  - Stratified train/val/test splitting
  - Z-score normalization (fit on train, apply to val/test)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PregnancyDataset(Dataset):
    """
    PyTorch Dataset for first-trimester pregnancy data.

    For each patient returns:
        - temporal_features : (T, 4) — FHR, CRL, GS, YSD per time step
        - gestational_ages  : (T,)   — actual GA weeks (for continuous PE)
        - temporal_mask     : (T,)   — 1.0 for real scans, 0.0 for padding
        - maternal_features : (7,)   — age, bmi, parity, gravidity,
                                        previous_loss, conception_ivf, singleton
        - label             : scalar — 0 or 1

    Parameters
    ----------
    patients_df : pd.DataFrame
        Patient-level data.
    scans_df : pd.DataFrame
        Scan-level data (multiple rows per patient).
    max_scans : int
        Maximum number of scans to keep per patient.
    us_features : list[str]
        Ultrasound feature column names.
    norm_stats : dict, optional
        Pre-computed {feature: (mean, std)} for normalization.
        If None, no normalization is applied.
    """

    US_FEATURES = ['FHR', 'CRL', 'GS', 'YSD']

    def __init__(
        self,
        patients_df: pd.DataFrame,
        scans_df: pd.DataFrame,
        max_scans: int = 5,
        us_features: Optional[List[str]] = None,
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.max_scans = max_scans
        self.us_features = us_features or self.US_FEATURES
        self.norm_stats = norm_stats

        # Store patient-level data
        self.patients = patients_df.copy().reset_index(drop=True)

        # One-hot encode conception method
        self.patients['conception_ivf'] = (
            self.patients['conception'].str.upper() == 'IVF'
        ).astype(float)
        self.patients['singleton'] = self.patients['singleton'].astype(float)

        # Group scans by patient (sorted by GA)
        scans_sorted = scans_df.sort_values(
            ['patient_id', 'gestational_age_weeks']
        ).copy()
        self.scans_by_patient: Dict[str, pd.DataFrame] = {
            pid: group for pid, group in scans_sorted.groupby('patient_id')
        }

        # Patient IDs in order
        self.patient_ids: List[str] = self.patients['patient_id'].tolist()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        pat_row = self.patients[self.patients['patient_id'] == pid].iloc[0]

        # ── Temporal features ─────────────────────────────────────────────
        scans = self.scans_by_patient.get(pid, pd.DataFrame())

        # Extract US measurements and GA
        if len(scans) > 0:
            us_vals = scans[self.us_features].values.astype(np.float32)
            ga_vals = scans['gestational_age_weeks'].values.astype(np.float32)
            n_real = min(len(scans), self.max_scans)
        else:
            us_vals = np.zeros((0, len(self.us_features)), dtype=np.float32)
            ga_vals = np.zeros((0,), dtype=np.float32)
            n_real = 0

        # Replace NaN with 0 (mask will handle it)
        us_vals = np.nan_to_num(us_vals, nan=0.0)

        # Apply normalization if stats are provided
        if self.norm_stats is not None:
            for i, feat in enumerate(self.us_features):
                if feat in self.norm_stats:
                    mean, std = self.norm_stats[feat]
                    if std > 1e-8:
                        us_vals[:, i] = (us_vals[:, i] - mean) / std

        # Pad / truncate to max_scans
        temporal_features = np.zeros(
            (self.max_scans, len(self.us_features)), dtype=np.float32
        )
        gestational_ages = np.zeros(self.max_scans, dtype=np.float32)
        temporal_mask = np.zeros(self.max_scans, dtype=np.float32)

        t = min(n_real, self.max_scans)
        if t > 0:
            temporal_features[:t] = us_vals[:t]
            gestational_ages[:t] = ga_vals[:t]
            temporal_mask[:t] = 1.0

        # ── Maternal features ─────────────────────────────────────────────
        mat_cols = ['age', 'bmi', 'parity', 'gravidity',
                    'previous_loss', 'conception_ivf', 'singleton']
        maternal = np.array(
            [float(pat_row[c]) for c in mat_cols], dtype=np.float32
        )

        # Normalize maternal features if stats provided
        if self.norm_stats is not None:
            for i, feat in enumerate(mat_cols):
                if feat in self.norm_stats:
                    mean, std = self.norm_stats[feat]
                    if std > 1e-8:
                        maternal[i] = (maternal[i] - mean) / std

        # ── Label ─────────────────────────────────────────────────────────
        label = float(pat_row['label'])

        return {
            'patient_id': pid,
            'temporal_features': torch.tensor(temporal_features),
            'gestational_ages': torch.tensor(gestational_ages),
            'temporal_mask': torch.tensor(temporal_mask),
            'maternal_features': torch.tensor(maternal),
            'label': torch.tensor(label),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────────────

def create_splits(
    patients_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified train / val / test split by label.

    Returns
    -------
    train_ids, val_ids, test_ids : lists of patient_id strings
    """
    rng = np.random.RandomState(seed)

    ids_0 = (
        patients_df[patients_df['label'] == 0]['patient_id']
        .to_numpy(dtype=str)
        .copy()
    )
    ids_1 = (
        patients_df[patients_df['label'] == 1]['patient_id']
        .to_numpy(dtype=str)
        .copy()
    )
    rng.shuffle(ids_0)
    rng.shuffle(ids_1)

    def _split(ids: np.ndarray):
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            ids[:n_train].tolist(),
            ids[n_train:n_train + n_val].tolist(),
            ids[n_train + n_val:].tolist(),
        )

    tr0, va0, te0 = _split(ids_0)
    tr1, va1, te1 = _split(ids_1)

    return tr0 + tr1, va0 + va1, te0 + te1


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(
    patients_df: pd.DataFrame,
    scans_df: pd.DataFrame,
    train_ids: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Z-score normalization statistics from the training split only.

    Returns dict mapping feature name → (mean, std).
    """
    train_scans = scans_df[scans_df['patient_id'].isin(train_ids)]
    train_patients = patients_df[patients_df['patient_id'].isin(train_ids)]

    stats: Dict[str, Tuple[float, float]] = {}

    # Ultrasound features
    for feat in ['FHR', 'CRL', 'GS', 'YSD']:
        vals = train_scans[feat].dropna()
        stats[feat] = (float(vals.mean()), float(vals.std()))

    # Maternal features (continuous only — skip binary features)
    for feat in ['age', 'bmi', 'parity', 'gravidity']:
        vals = train_patients[feat].dropna()
        stats[feat] = (float(vals.mean()), float(vals.std()))

    # Binary features: don't normalize (keep 0/1)
    # previous_loss, conception_ivf, singleton — NOT in stats

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that handles patient_id strings."""
    return {
        'patient_id': [b['patient_id'] for b in batch],
        'temporal_features': torch.stack([b['temporal_features'] for b in batch]),
        'gestational_ages': torch.stack([b['gestational_ages'] for b in batch]),
        'temporal_mask': torch.stack([b['temporal_mask'] for b in batch]),
        'maternal_features': torch.stack([b['maternal_features'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


def get_dataloaders(
    patients_csv: str,
    scans_csv: str,
    batch_size: int = 32,
    max_scans: int = 5,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, Tuple[float, float]]]:
    """
    End-to-end pipeline: load CSVs → split → normalize → create DataLoaders.

    Returns
    -------
    loaders : dict with keys 'train', 'val', 'test'
    norm_stats : dict mapping feature → (mean, std), fit on train only
    """
    patients_df = pd.read_csv(patients_csv)
    scans_df = pd.read_csv(scans_csv)

    # Stratified split
    train_ids, val_ids, test_ids = create_splits(patients_df, seed=seed)

    # Compute normalization stats from training data only
    norm_stats = compute_norm_stats(patients_df, scans_df, train_ids)

    loaders = {}
    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        subset_patients = patients_df[patients_df['patient_id'].isin(ids)]
        subset_scans = scans_df[scans_df['patient_id'].isin(ids)]

        ds = PregnancyDataset(
            patients_df=subset_patients,
            scans_df=subset_scans,
            max_scans=max_scans,
            norm_stats=norm_stats,
        )

        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == 'train'),
            num_workers=num_workers,
            collate_fn=_collate_fn,
            drop_last=False,
        )

    print(f"Splits — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Normalization stats: { {k: (f'{v[0]:.2f}', f'{v[1]:.2f}') for k, v in norm_stats.items()} }")

    return loaders, norm_stats
