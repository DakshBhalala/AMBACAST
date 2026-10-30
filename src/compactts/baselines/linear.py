"""Compact Linear Forecasters: DLinear, NLinear, and Ridge Regression.

References:
    Zeng et al., "Are Transformers Effective for Time Series?", AAAI 2023.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from compactts.models.components import MovingAverage


class RidgeForecaster:
    """Multi-output Ridge regression predicting future horizon directly from context."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeForecaster":
        """Fit Ridge on windowed (N, context_len) -> (N, pred_len)."""
        if X.ndim == 3:
            N, L, C = X.shape
            X_2d = X.reshape(N, L * C)
            Y_2d = Y.reshape(N, -1)
        else:
            X_2d = X
            Y_2d = Y
        self.model.fit(X_2d, Y_2d)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from context windows."""
        orig_ndim = X.ndim
        if orig_ndim == 3:
            N, L, C = X.shape
            X_2d = X.reshape(N, L * C)
            preds = self.model.predict(X_2d)
            return preds.reshape(N, -1, C)
        else:
            return self.model.predict(X)


class DLinear(nn.Module):
    """DLinear baseline: Moving average decomposition with two separate linear layers."""

    def __init__(self, context_len: int, pred_len: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.decompsition = MovingAverage(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(context_len, pred_len)
        self.linear_trend = nn.Linear(context_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, context_len, channels) or (batch, context_len)
        """
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            was_2d = True

        trend_init = self.decompsition(x)
        seasonal_init = x - trend_init

        # Permute to (batch, channels, context_len) for Linear
        trend_perm = trend_init.permute(0, 2, 1)
        seasonal_perm = seasonal_init.permute(0, 2, 1)

        trend_output = self.linear_trend(trend_perm)
        seasonal_output = self.linear_seasonal(seasonal_perm)

        out = trend_output + seasonal_output
        out = out.permute(0, 2, 1)  # (batch, pred_len, channels)

        if was_2d:
            out = out.squeeze(-1)
        return out


class NLinear(nn.Module):
    """NLinear baseline: Normalization by subtracting last value, single linear projection, add back."""

    def __init__(self, context_len: int, pred_len: int) -> None:
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.linear = nn.Linear(context_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            was_2d = True

        seq_last = x[:, -1:, :].detach()
        x_norm = x - seq_last

        x_perm = x_norm.permute(0, 2, 1)
        out = self.linear(x_perm)
        out = out.permute(0, 2, 1)

        out = out + seq_last
        if was_2d:
            out = out.squeeze(-1)
        return out
