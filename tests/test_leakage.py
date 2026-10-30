"""Tests for data leakage detection and anti-leakage audit suite."""

import numpy as np
import pytest
from compactts.evaluation.leakage_checker import LeakageChecker
from compactts.data.transforms import SafeStandardScaler


def test_chronological_order_detection():
    # Clean non-overlapping sequential indices
    train_idx = np.arange(0, 70)
    val_idx = np.arange(70, 85)
    test_idx = np.arange(85, 100)

    clean_res = LeakageChecker.verify_chronological_order(train_idx, val_idx, test_idx)
    assert clean_res["is_leakage_free"] is True

    # Corrupted / Leaking indices (overlapping)
    leaking_val_idx = np.arange(65, 85)
    leak_res = LeakageChecker.verify_chronological_order(train_idx, leaking_val_idx, test_idx)
    assert leak_res["is_leakage_free"] is False


def test_scaler_leakage_detection():
    train_data = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    full_data = np.array([[10.0], [20.0], [30.0], [100.0], [200.0]], dtype=np.float32)

    # Valid scaler fitted only on train
    scaler_clean = SafeStandardScaler()
    scaler_clean.fit(train_data)
    clean_audit = LeakageChecker.verify_scaler_integrity(train_data, full_data, scaler_clean)
    assert clean_audit["is_leakage_free"] is True

    # Leaking scaler fitted on full_data (including future test points)
    scaler_leaking = SafeStandardScaler()
    scaler_leaking.fit(full_data)
    leak_audit = LeakageChecker.verify_scaler_integrity(train_data, full_data, scaler_leaking)
    assert leak_audit["is_leakage_free"] is False


def test_window_boundary_detection():
    # Clean boundary
    res_clean = LeakageChecker.verify_window_boundary(train_target_end_idx=70, test_context_start_idx=70)
    assert res_clean["is_leakage_free"] is True

    # Leaking boundary: train target reaches 75 while test starts at 70
    res_leak = LeakageChecker.verify_window_boundary(train_target_end_idx=75, test_context_start_idx=70)
    assert res_leak["is_leakage_free"] is False
