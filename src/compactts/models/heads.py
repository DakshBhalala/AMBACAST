"""Forecasting prediction heads for CompactTS.

Supports:
- Deterministic point forecasting
- Multi-quantile probabilistic forecasting
- Parametric Gaussian distribution forecasting
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class PointHead(nn.Module):
    """Deterministic point forecast head."""

    def __init__(self, in_features: int, pred_len: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, pred_len),
            )
        else:
            self.net = nn.Linear(in_features, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to forecast horizon.

        Args:
            x: Tensor of shape (batch, channels, in_features).

        Returns:
            Tensor of shape (batch, pred_len, channels).
        """
        # (batch, channels, pred_len)
        out = self.net(x)
        # Permute to standard (batch, pred_len, channels)
        return out.permute(0, 2, 1)


class QuantileHead(nn.Module):
    """Multi-quantile probabilistic forecast head."""

    def __init__(
        self,
        in_features: int,
        pred_len: int,
        quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        if hidden_dim is not None and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, pred_len * self.num_quantiles),
            )
        else:
            self.net = nn.Linear(in_features, pred_len * self.num_quantiles)

        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to multi-quantile forecast.

        Args:
            x: Tensor of shape (batch, channels, in_features).

        Returns:
            Tensor of shape (batch, pred_len, channels, num_quantiles).
        """
        batch, channels, _ = x.shape
        # (batch, channels, pred_len * num_quantiles)
        h = self.net(x)
        # Reshape to (batch, channels, pred_len, num_quantiles)
        h = h.view(batch, channels, self.pred_len, self.num_quantiles)
        # Permute to (batch, pred_len, channels, num_quantiles)
        out = h.permute(0, 2, 1, 3)
        return out


class DistributionHead(nn.Module):
    """Parametric Gaussian probabilistic forecast head."""

    def __init__(self, in_features: int, pred_len: int) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(in_features, pred_len)
        self.logvar_layer = nn.Linear(in_features, pred_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project to Gaussian parameters.

        Args:
            x: Tensor of shape (batch, channels, in_features).

        Returns:
            (mean, std) both of shape (batch, pred_len, channels).
        """
        mean = self.mean_layer(x).permute(0, 2, 1)
        logvar = self.logvar_layer(x).permute(0, 2, 1)
        # Numerical clamping for stable variance
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        std = torch.exp(0.5 * logvar)
        return mean, std
