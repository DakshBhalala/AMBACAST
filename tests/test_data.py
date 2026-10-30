"""Tests for dataset windowing, imputation, and chronological splitting."""

import numpy as np
import pytest
from compactts.data.dataset import (
    chronological_split,
    create_sliding_windows,
    clean_and_impute_series,
    TimeSeriesDataset,
)


def test_clean_and_impute():
    arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    cleaned = clean_and_impute_series(arr, method="linear")
    assert not np.any(np.isnan(cleaned))
    assert np.allclose(cleaned[1], 2.0)
    assert np.allclose(cleaned[3], 4.0)


def test_chronological_split():
    data = np.arange(100, dtype=np.float32)
    train, val, test = chronological_split(data, 0.7, 0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Strict chronological monotonicity
    assert train[-1] < val[0]
    assert val[-1] < test[0]


def test_create_sliding_windows():
    data = np.arange(100, dtype=np.float32)
    X, Y = create_sliding_windows(data, context_len=20, pred_len=10, stride=2)
    # (100 - 30) // 2 + 1 = 70 // 2 + 1 = 36 windows
    assert X.shape == (36, 20, 1)
    assert Y.shape == (36, 10, 1)

    # Causality test: Y[i, 0] must immediately follow X[i, -1]
    for i in range(len(X)):
        assert Y[i, 0, 0] == X[i, -1, 0] + 1


def test_sliding_windows_too_short():
    data = np.arange(10, dtype=np.float32)
    with pytest.raises(ValueError):
        create_sliding_windows(data, context_len=10, pred_len=5)


def test_timeseries_dataset():
    X = np.random.randn(30, 24, 2).astype(np.float32)
    Y = np.random.randn(30, 6, 2).astype(np.float32)
    ds = TimeSeriesDataset(X, Y)
    assert len(ds) == 30
    x_item, y_item = ds[0]
    assert x_item.shape == (24, 2)
    assert y_item.shape == (6, 2)
