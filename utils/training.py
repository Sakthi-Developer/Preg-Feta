"""
Training utilities for the AVM pregnancy loss prediction models.

Provides:
  - :class:`Trainer` — full training loop with validation, early stopping,
    checkpointing, cosine-annealing LR scheduling, and per-epoch logging.
  - :class:`EarlyStopping` — monitors a metric and triggers stopping.
  - :func:`compute_class_weights` — derives ``pos_weight`` for BCE loss.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from data.graph_builder import PatientGraphBuilder
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def get_best_device() -> torch.device:
    """Auto-detect the best available compute device (CUDA > MPS > CPU).

    Returns:
        :class:`torch.device` for the fastest available backend.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Class weight helper
# ---------------------------------------------------------------------------

def compute_class_weights(labels: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute ``pos_weight`` for BCEWithLogitsLoss from label distribution.

    The weight compensates for class imbalance so that the minority
    (positive) class contributes equally to the loss:

        pos_weight = n_negative / n_positive

    Args:
        labels: 1-D array/tensor of binary labels (0 or 1).

    Returns:
        Scalar ``pos_weight`` value.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        logger.warning("No positive samples found; returning pos_weight=1.0")
        return 1.0
    return float(n_neg / n_pos)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping monitor.

    Tracks a validation metric and triggers stopping when the metric does
    not improve for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``'max'`` (higher is better, e.g. AUROC) or ``'min'``
              (lower is better, e.g. loss).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return current > self.best_score + self.min_delta
        return current < self.best_score - self.min_delta

    def __call__(self, metric_value: float) -> bool:
        """Update state with the latest metric value.

        Args:
            metric_value: The monitored metric for this epoch.

        Returns:
            ``True`` if training should stop.
        """
        if self._is_improvement(metric_value):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop with validation, early stopping, and checkpointing.

    Handles:
      - Train / validate epoch loops
      - Weighted BCE loss
      - Cosine annealing with warm restarts
      - Early stopping by validation AUROC
      - Model checkpointing (best val AUROC)
      - Per-epoch metric logging (loss, AUROC, AUPRC)

    Args:
        model: The ``nn.Module`` to train.
        train_loader: Training ``DataLoader``.
        val_loader: Validation ``DataLoader``.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
        pos_weight: Positive class weight for ``BCEWithLogitsLoss``.
        patience: Early stopping patience (epochs).
        checkpoint_dir: Directory to save model checkpoints.
        device: Override device (default: auto-detect).
        t_0: Period for the first warm restart (cosine annealing).
        t_mult: Multiplicative factor for subsequent restarts.
        collate_mode: ``'feta'``, ``'preg'``, or ``'ensemble'`` — controls
            how each batch dict is unpacked. Defaults to ``'ensemble'``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None,
        t_0: int = 10,
        t_mult: int = 2,
        collate_mode: str = "ensemble",
        graph_builder: Optional[PatientGraphBuilder] = None,
        max_scans: int = 5,
    ) -> None:
        self.device = device or get_best_device()
        logger.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        if collate_mode not in {"feta", "preg", "ensemble"}:
            raise ValueError("collate_mode must be 'feta', 'preg', or 'ensemble'")
        self.collate_mode = collate_mode
        self.graph_builder = graph_builder or PatientGraphBuilder(max_scans=max_scans)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )

        # Optimizer & scheduler
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t_0, T_mult=t_mult
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_auroc: float = 0.0

        # History
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Batch unpacking helpers
    # ------------------------------------------------------------------ #

    def _unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Move batch tensors to device and split inputs from labels.

        Returns:
            (inputs_dict, labels) where ``inputs_dict`` can be
            unpacked as ``**inputs_dict`` into the model's forward.
        """
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if "label" in batch:
            labels = batch.pop("label")
        elif "labels" in batch:
            labels = batch.pop("labels")
        else:
            raise KeyError("Batch must contain 'label' or 'labels'.")

        batch.pop("patient_id", None)

        if self.collate_mode == "feta":
            inputs = {
                "temporal_features": batch["temporal_features"],
                "gestational_ages": batch["gestational_ages"],
                "temporal_mask": batch["temporal_mask"],
                "maternal_features": batch["maternal_features"],
            }
            return inputs, labels

        graph = self.graph_builder.batch_to_graph(batch)
        graph = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in graph.items()
        }
        graph_inputs = {
            "node_features": graph["node_features"],
            "edge_index": graph["edge_index"],
            "node_types": graph["node_types"],
            "node_mask": graph["node_mask"],
        }

        if self.collate_mode == "preg":
            return graph_inputs, labels

        inputs = {
            "temporal_features": batch["temporal_features"],
            "gestational_ages": batch["gestational_ages"],
            "temporal_mask": batch["temporal_mask"],
            "maternal_features": batch["maternal_features"],
            **graph_inputs,
        }
        return inputs, labels

    # ------------------------------------------------------------------ #
    # Train / Validate
    # ------------------------------------------------------------------ #

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict with ``loss``, ``auroc``, ``auprc``.
        """
        self.model.train()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            inputs, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)

            # Handle both dict outputs (ensemble) and tuple outputs
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits.squeeze(-1), labels.float())
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        self.scheduler.step()

        # Compute epoch metrics
        all_logits_t = torch.cat(all_logits).squeeze(-1)
        all_labels_t = torch.cat(all_labels)
        y_prob = torch.sigmoid(all_logits_t).numpy()
        y_true = all_labels_t.numpy()
        metrics = compute_all_metrics(y_true, y_prob)

        return {
            "loss": total_loss / max(n_batches, 1),
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch.

        Returns:
            Dict with ``loss``, ``auroc``, ``auprc``, and all other metrics.
        """
        self.model.eval()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            inputs, labels = self._unpack_batch(batch)
            outputs = self.model(**inputs)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits.squeeze(-1), labels.float())
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits_t = torch.cat(all_logits).squeeze(-1)
        all_labels_t = torch.cat(all_labels)
        y_prob = torch.sigmoid(all_logits_t).numpy()
        y_true = all_labels_t.numpy()
        metrics = compute_all_metrics(y_true, y_prob)
        metrics["loss"] = total_loss / max(n_batches, 1)

        return metrics

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, auroc: float) -> str:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            auroc: Validation AUROC at this epoch.

        Returns:
            Path to the saved checkpoint file.
        """
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_auroc": auroc,
            },
            path,
        )
        logger.info("Saved checkpoint at epoch %d (AUROC=%.4f) -> %s", epoch, auroc, path)
        return str(path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint and restore model / optimizer / scheduler state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The full checkpoint dict.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_auroc = checkpoint.get("best_auroc", 0.0)
        logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint["epoch"])
        return checkpoint

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def fit(self, n_epochs: int = 100) -> List[Dict[str, Any]]:
        """Run the full training loop.

        Args:
            n_epochs: Maximum number of training epochs.

        Returns:
            List of per-epoch metric dicts (the training history).
        """
        logger.info("Starting training for up to %d epochs", n_epochs)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            elapsed = time.time() - t0

            # Log
            record = {
                "epoch": epoch,
                "time_s": elapsed,
                "train_loss": train_metrics["loss"],
                "train_auroc": train_metrics["auroc"],
                "train_auprc": train_metrics["auprc"],
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
            }
            self.history.append(record)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f | "
                "val_AUROC=%.4f  val_AUPRC=%.4f | %.1fs",
                epoch,
                n_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["auroc"],
                val_metrics["auprc"],
                elapsed,
            )

            # Checkpoint best model
            val_auroc = val_metrics["auroc"]
            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
                self._save_checkpoint(epoch, val_auroc)

            # Early stopping
            if self.early_stopping(val_auroc):
                logger.info(
                    "Early stopping triggered at epoch %d (best AUROC=%.4f)",
                    epoch,
                    self.best_auroc,
                )
                break

        return self.history
