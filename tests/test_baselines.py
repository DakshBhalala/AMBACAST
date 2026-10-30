"""Tests for statistical, linear, and GBDT baselines."""

import numpy as np
import torch
import pytest
from compactts.baselines.statistical import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ExponentialSmoothingForecaster,
    AutoregressiveForecaster,
)
from compactts.baselines.linear import RidgeForecaster, DLinear, NLinear
from compactts.baselines.gbm import LightGBMForecaster


def test_naive_forecaster():
    y = np.array([1.0, 2.0, 5.0])
    naive = NaiveForecaster()
    naive.fit(y)
    preds = naive.predict(pred_len=4)
    assert np.allclose(preds, np.array([5.0, 5.0, 5.0, 5.0]))


def test_seasonal_naive_forecaster():
    y = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    snaive = SeasonalNaiveForecaster(season_length=3)
    snaive.fit(y)
    preds = snaive.predict(pred_len=5)
    assert np.allclose(preds, np.array([10.0, 20.0, 30.0, 10.0, 20.0]))


def test_exponential_smoothing_forecaster():
    t = np.arange(100, dtype=np.float32)
    y = 0.5 * t + 3.0 * np.sin(2 * np.pi * t / 12)
    ets = ExponentialSmoothingForecaster(season_length=12)
    ets.fit(y)
    preds = ets.predict(pred_len=12)
    assert len(preds) == 12
    assert not np.any(np.isnan(preds))


def test_autoregressive_forecaster():
    t = np.arange(80, dtype=np.float32)
    y = np.sin(2 * np.pi * t / 10)
    ar = AutoregressiveForecaster(p=5)
    ar.fit(y)
    preds = ar.predict(pred_len=10)
    assert len(preds) == 10
    assert not np.any(np.isnan(preds))


def test_ridge_forecaster():
    X = np.random.randn(50, 20).astype(np.float32)
    Y = np.random.randn(50, 5).astype(np.float32)
    ridge = RidgeForecaster(alpha=0.5)
    ridge.fit(X, Y)
    preds = ridge.predict(X[:5])
    assert preds.shape == (5, 5)


def test_dlinear_and_nlinear():
    x = torch.randn(4, 32, 1)
    dlinear = DLinear(context_len=32, pred_len=8)
    nlinear = NLinear(context_len=32, pred_len=8)
    out_d = dlinear(x)
    out_n = nlinear(x)
    assert out_d.shape == (4, 8, 1)
    assert out_n.shape == (4, 8, 1)


def test_lightgbm_forecaster():
    X = np.random.randn(60, 24).astype(np.float32)
    Y = np.random.randn(60, 6).astype(np.float32)
    lgbm = LightGBMForecaster(n_estimators=10, max_depth=3)
    lgbm.fit(X, Y)
    preds = lgbm.predict(X[:3])
    assert preds.shape == (3, 6)
