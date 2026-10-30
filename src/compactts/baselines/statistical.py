"""Statistical baselines for time-series forecasting.

Includes:
- Naive Forecaster
- Seasonal Naive Forecaster
- Exponential Smoothing (Holt-Winters / AutoETS)
- Autoregressive AR(p) Forecaster
"""

from typing import Optional, Union, Tuple
import numpy as np
from scipy.optimize import minimize


class NaiveForecaster:
    """Naive forecaster: predicts the last observed value for all future horizons."""

    def __init__(self) -> None:
        self.last_value: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        """Fit on history.

        Args:
            y: Array of shape (T, C) or (T,).
        """
        if y.ndim == 1:
            self.last_value = y[-1:]
        else:
            self.last_value = y[-1:, :]
        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Predict future values.

        Returns:
            Array of shape (pred_len, C) or (pred_len,).
        """
        if self.last_value is None:
            raise RuntimeError("NaiveForecaster must be fit before predict.")
        return np.repeat(self.last_value, pred_len, axis=0)


class SeasonalNaiveForecaster:
    """Seasonal Naive forecaster: repeats values from the last seasonal cycle.

    If pred_len > season_length, tiles the seasonal cycle.
    """

    def __init__(self, season_length: int = 24) -> None:
        self.season_length = season_length
        self.history: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        """Fit on history.

        Args:
            y: Array of shape (T, C) or (T,).
        """
        if len(y) < self.season_length:
            # Fall back to repeating whatever we have
            self.season_length = max(1, len(y))
        self.history = y[-self.season_length:]
        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Predict future values.

        Returns:
            Array of shape (pred_len, C) or (pred_len,).
        """
        if self.history is None:
            raise RuntimeError("SeasonalNaiveForecaster must be fit before predict.")
        repeats = int(np.ceil(pred_len / self.season_length))
        tiled = np.tile(self.history, (repeats, 1) if self.history.ndim > 1 else (repeats,))
        return tiled[:pred_len]


class ExponentialSmoothingForecaster:
    """Holt-Winters Exponential Smoothing (Level + Trend + Seasonality) with parameter fitting."""

    def __init__(self, season_length: int = 24, seasonal: Optional[str] = "additive") -> None:
        self.season_length = season_length
        self.seasonal = seasonal
        self.alpha = 0.3
        self.beta = 0.1
        self.gamma = 0.1
        self.level = 0.0
        self.trend = 0.0
        self.seasonals: np.ndarray = np.array([])

    def fit(self, y: np.ndarray) -> "ExponentialSmoothingForecaster":
        """Fit smoothing parameters on univariate series."""
        y_flat = y.ravel().astype(np.float64)
        n = len(y_flat)
        s_len = min(self.season_length, n // 2) if n > 4 else 1

        if s_len <= 1 or self.seasonal is None:
            # Simple / Double exponential smoothing
            def loss(params: np.ndarray) -> float:
                a, b = params
                lvl = y_flat[0]
                trd = (y_flat[1] - y_flat[0]) if n > 1 else 0.0
                err = 0.0
                for t in range(n):
                    y_pred = lvl + trd
                    err += (y_flat[t] - y_pred) ** 2
                    new_lvl = a * y_flat[t] + (1 - a) * (lvl + trd)
                    trd = b * (new_lvl - lvl) + (1 - b) * trd
                    lvl = new_lvl
                return err / n

            res = minimize(loss, [0.3, 0.1], bounds=[(0.01, 0.99), (0.01, 0.99)], method="L-BFGS-B")
            self.alpha, self.beta = res.x
            # Run forward pass to get final states
            self.level = y_flat[0]
            self.trend = (y_flat[1] - y_flat[0]) if n > 1 else 0.0
            for t in range(n):
                new_lvl = self.alpha * y_flat[t] + (1 - self.alpha) * (self.level + self.trend)
                self.trend = self.beta * (new_lvl - self.level) + (1 - self.beta) * self.trend
                self.level = new_lvl
            self.seasonals = np.zeros(1)
        else:
            # Additive Holt-Winters
            self.season_length = s_len
            self.level = np.mean(y_flat[:s_len])
            self.trend = (y_flat[s_len] - y_flat[0]) / s_len if n > s_len else 0.0
            self.seasonals = y_flat[:s_len] - self.level

            def loss_hw(params: np.ndarray) -> float:
                a, b, g = params
                lvl = np.mean(y_flat[:s_len])
                trd = (y_flat[s_len] - y_flat[0]) / s_len if n > s_len else 0.0
                seas = list(y_flat[:s_len] - lvl)
                err = 0.0
                for t in range(s_len, n):
                    s_idx = t % s_len
                    y_pred = lvl + trd + seas[s_idx]
                    err += (y_flat[t] - y_pred) ** 2
                    new_lvl = a * (y_flat[t] - seas[s_idx]) + (1 - a) * (lvl + trd)
                    trd = b * (new_lvl - lvl) + (1 - b) * trd
                    seas[s_idx] = g * (y_flat[t] - new_lvl) + (1 - g) * seas[s_idx]
                    lvl = new_lvl
                return err / max(1, n - s_len)

            res = minimize(loss_hw, [0.3, 0.1, 0.1], bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)], method="L-BFGS-B")
            self.alpha, self.beta, self.gamma = res.x

            # Compute final state
            lvl = np.mean(y_flat[:s_len])
            trd = (y_flat[s_len] - y_flat[0]) / s_len if n > s_len else 0.0
            seas = list(y_flat[:s_len] - lvl)
            for t in range(s_len, n):
                s_idx = t % s_len
                new_lvl = self.alpha * (y_flat[t] - seas[s_idx]) + (1 - self.alpha) * (lvl + trd)
                trd = self.beta * (new_lvl - lvl) + (1 - self.beta) * trd
                seas[s_idx] = self.gamma * (y_flat[t] - new_lvl) + (1 - self.gamma) * seas[s_idx]
                lvl = new_lvl
            self.level = lvl
            self.trend = trd
            self.seasonals = np.array(seas)

        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Predict pred_len steps into future."""
        preds = np.zeros(pred_len, dtype=np.float32)
        s_len = len(self.seasonals)
        for h in range(1, pred_len + 1):
            if s_len > 1:
                s_val = self.seasonals[(h - 1) % s_len]
            else:
                s_val = 0.0
            preds[h - 1] = self.level + h * self.trend + s_val
        return preds


class AutoregressiveForecaster:
    """Autoregressive AR(p) forecaster fitted with Ordinary Least Squares."""

    def __init__(self, p: int = 24) -> None:
        self.p = p
        self.coeffs: Optional[np.ndarray] = None
        self.intercept = 0.0
        self.last_p_values: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "AutoregressiveForecaster":
        """Fit AR(p) coefficients using ridge-regularized OLS."""
        y_flat = y.ravel().astype(np.float64)
        n = len(y_flat)
        actual_p = min(self.p, n // 3) if n > 6 else 1
        self.p = actual_p

        # Construct lagged matrix
        X = np.zeros((n - actual_p, actual_p), dtype=np.float64)
        Y = y_flat[actual_p:]
        for i in range(actual_p):
            X[:, i] = y_flat[actual_p - i - 1 : n - i - 1]

        # Add intercept
        X_design = np.hstack([np.ones((len(X), 1)), X])
        # Regularized solve (X^T X + lambda I)^-1 X^T Y
        reg = 1e-3 * np.eye(X_design.shape[1])
        reg[0, 0] = 0.0  # Do not regularize intercept
        params = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ Y)

        self.intercept = params[0]
        self.coeffs = params[1:]
        self.last_p_values = y_flat[-actual_p:][::-1]  # Most recent first
        return self

    def predict(self, pred_len: int) -> np.ndarray:
        """Predict autoregressively for pred_len steps."""
        if self.coeffs is None or self.last_p_values is None:
            raise RuntimeError("AutoregressiveForecaster must be fit before predict.")

        history = list(self.last_p_values)
        preds = []
        for _ in range(pred_len):
            val = self.intercept + np.dot(self.coeffs, history[: self.p])
            preds.append(val)
            history.insert(0, val)
        return np.array(preds, dtype=np.float32)
