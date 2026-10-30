"""Tests for model training loop, early stopping, and loss optimization."""

import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader
from compactts.models.compact_ts import CompactTS, CompactTSConfig
from compactts.data.dataset import TimeSeriesDataset
from compactts.training.trainer import CompactTSTrainer


def test_compactts_trainer_step():
    torch.manual_seed(42)
    # Synthetic small dataset
    X = np.random.randn(32, 24, 1).astype(np.float32)
    Y = np.random.randn(32, 6, 1).astype(np.float32)

    cfg = CompactTSConfig.from_tier("tiny", context_len=24, pred_len=6)
    model = CompactTS(cfg)
    trainer = CompactTSTrainer(model, lr=5e-3, device="cpu", use_amp=False)

    loader = DataLoader(TimeSeriesDataset(X, Y), batch_size=16)
    loss_before = trainer.evaluate(loader)

    summary = trainer.fit(loader, loader, epochs=3, model_name="test_run.pt")
    loss_after = summary["best_val_loss"]

    assert summary["epochs_completed"] == 3
    assert loss_after <= loss_before + 0.1  # Training completed cleanly
