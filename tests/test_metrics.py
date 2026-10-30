"""Tests for forecasting evaluation metrics."""

import numpy as np
import pytest
from compactts.evaluation.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_scaled_error,
    weighted_quantile_loss,
    continuous_ranked_probability_score,
    calculate_metrics,
)


def test_standard_metrics():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 4.0, 9.0])

    assert np.isclose(mean_squared_error(y_true, y_pred), (1 + 0 + 9) / 3)
    assert np.isclose(mean_absolute_error(y_true, y_pred), (1 + 0 + 3) / 3)
    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt((1 + 0 + 9) / 3))


def test_mase_calculation():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.5, 7.5])  # MAE = 0.5
    # y_train diff is [1, 1, 1, 1], mean diff = 1.0
    # MASE = 0.5 / 1.0 = 0.5
    mase = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=1)
    assert np.isclose(mase, 0.5)


def test_wql_and_crps():
    y_true = np.array([[2.0], [4.0]])  # (2, 1)
    # Perfect quantile predictions
    q_preds = np.array([
        [[1.5, 2.0, 2.5]],
        [[3.5, 4.0, 4.5]],
    ])  # (2, 1, 3)
    quantiles = (0.1, 0.5, 0.9)
    wql = weighted_quantile_loss(y_true, q_preds, quantiles)
    crps = continuous_ranked_probability_score(y_true, q_preds, quantiles)

    assert wql >= 0.0
    assert crps >= 0.0


def test_calculate_metrics_dict():
    y_train = np.arange(20, dtype=np.float32)
    y_true = np.array([20.0, 21.0])
    y_pred = np.array([20.2, 21.1])
    res = calculate_metrics(y_true, y_pred, y_train=y_train)

    assert "mse" in res
    assert "mae" in res
    assert "rmse" in res
    assert "mase" in res
    assert "smape" in res
