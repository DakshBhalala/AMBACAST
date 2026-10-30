"""Evaluation metrics for time-series forecasting benchmarks (GIFT-Eval compliant).

Includes:
- MAE, MSE, RMSE, MAPE, sMAPE
- MASE (Mean Absolute Scaled Error, relative to in-sample seasonal naive)
- WQL (Weighted Quantile Loss)
- CRPS (Continuous Ranked Probability Score for quantiles)
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (MAPE)."""
    denom = np.abs(y_true) + eps
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def symmetric_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    Scales MAE by in-sample one-step or seasonal naive error:
        scale = mean(|y_train[t] - y_train[t - seasonality]|)

    Args:
        y_true: Test target values.
        y_pred: Model predictions.
        y_train: In-sample training series.
        seasonality: Seasonal period (default 1 for non-seasonal naive).
        eps: Small constant to prevent division by zero.
    """
    mae = np.mean(np.abs(y_true - y_pred))

    if len(y_train) <= seasonality:
        scale = np.mean(np.abs(np.diff(y_train, axis=0))) if len(y_train) > 1 else 1.0
    else:
        diff = y_train[seasonality:] - y_train[:-seasonality]
        scale = np.mean(np.abs(diff))

    if scale < eps:
        scale = 1.0

    return float(mae / scale)


def weighted_quantile_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    eps: float = 1e-8,
) -> float:
    """Weighted Quantile Loss (WQL).

    For each quantile q:
        QL_q = 2 * sum( max(q * (y - y_hat), (1 - q) * (y_hat - y)) )
    WQL = mean_q( QL_q / sum(|y|) )

    Args:
        y_true: Target values of shape (N, pred_len) or (N, pred_len, C).
        quantile_preds: Forecasts of shape (N, pred_len, num_quantiles) or (N, pred_len, C, num_quantiles).
        quantiles: Tuple of quantile levels.
    """
    if y_true.ndim == quantile_preds.ndim - 1:
        # Match dimensions by expanding y_true
        y_expanded = np.expand_dims(y_true, axis=-1)
    else:
        y_expanded = y_true

    denom = np.sum(np.abs(y_true)) + eps
    q_losses = []

    for idx, q in enumerate(quantiles):
        pred_q = quantile_preds[..., idx]
        target = y_expanded[..., 0] if y_expanded.shape[-1] == 1 else y_true
        error = target - pred_q
        loss_q = 2.0 * np.sum(np.maximum(q * error, (q - 1.0) * error))
        q_losses.append(loss_q / denom)

    return float(np.mean(q_losses))


def continuous_ranked_probability_score(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """Continuous Ranked Probability Score (CRPS) computed via quantile discretization.

    Approximated by numerical trapezoidal integration of pinball loss across quantiles:
        CRPS = 2 * int_0^1 Pinball_q(y, q) dq
    """
    if y_true.ndim == quantile_preds.ndim - 1:
        y_expanded = np.expand_dims(y_true, axis=-1)
    else:
        y_expanded = y_true

    target = y_expanded[..., 0] if y_expanded.shape[-1] == 1 else y_true
    pinball_losses = []

    for idx, q in enumerate(quantiles):
        pred_q = quantile_preds[..., idx]
        err = target - pred_q
        loss_q = np.mean(np.maximum(q * err, (q - 1.0) * err))
        pinball_losses.append(loss_q)

    # 2 * mean pinball loss is standard discrete quantile CRPS estimator
    return float(2.0 * np.mean(pinball_losses))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 1,
    quantile_preds: Optional[np.ndarray] = None,
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics dictionary."""
    # Squeeze extra unit trailing dimensions if shapes mismatch
    if y_true.ndim != y_pred.ndim:
        if y_true.ndim == y_pred.ndim + 1 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        elif y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)

    results: Dict[str, float] = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mape(y_true, y_pred),
    }

    if y_train is not None:
        results["mase"] = mean_absolute_scaled_error(
            y_true=y_true,
            y_pred=y_pred,
            y_train=y_train,
            seasonality=seasonality,
        )

    if quantile_preds is not None:
        results["wql"] = weighted_quantile_loss(
            y_true=y_true,
            quantile_preds=quantile_preds,
            quantiles=quantiles,
        )
        results["crps"] = continuous_ranked_probability_score(
            y_true=y_true,
            quantile_preds=quantile_preds,
            quantiles=quantiles,
        )

    return results
