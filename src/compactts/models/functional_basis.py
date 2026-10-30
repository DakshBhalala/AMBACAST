"""Orthogonal Functional Basis Decoder for Smooth, Oscillation-Free Forecasting.

Inspired by FlowState (IBM Research) and N-BEATS / N-HiTS:
Decodes latent representations into Chebyshev polynomial trend coefficients
and Fourier harmonic seasonality coefficients, ensuring guaranteed smoothness
and zero boundary oscillation across arbitrary forecast horizons.
"""

from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn


class FunctionalBasisDecoder(nn.Module):
    """Decodes latent features into Chebyshev polynomial and Fourier harmonic bases."""

    def __init__(
        self,
        in_features: int,
        pred_len: int,
        channels: int = 1,
        max_polynomial_degree: int = 4,
        num_harmonics: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.pred_len = pred_len
        self.channels = channels
        self.max_degree = max_polynomial_degree
        self.num_harmonics = min(num_harmonics, pred_len // 2)

        # 1. Precompute Chebyshev Polynomial Basis Matrix (pred_len, max_degree + 1)
        # Normalized time domain: tau in [-1, 1]
        t = np.linspace(-1.0, 1.0, pred_len, dtype=np.float32)
        cheb_basis = np.zeros((pred_len, self.max_degree + 1), dtype=np.float32)
        for d in range(self.max_degree + 1):
            if d == 0:
                cheb_basis[:, d] = 1.0
            elif d == 1:
                cheb_basis[:, d] = t
            else:
                cheb_basis[:, d] = 2.0 * t * cheb_basis[:, d - 1] - cheb_basis[:, d - 2]
        self.register_buffer("basis_trend", torch.from_numpy(cheb_basis), persistent=False)  # (H, P+1)
        self.register_buffer("basis_trend_t", torch.from_numpy(cheb_basis).t().contiguous(), persistent=False)

        # 2. Precompute Fourier Harmonic Basis Matrix (pred_len, 2 * num_harmonics)
        # Normalized time domain: 0 to 1
        t_fourier = np.linspace(0.0, 1.0, pred_len, endpoint=False, dtype=np.float32)
        fourier_basis = np.zeros((pred_len, 2 * self.num_harmonics), dtype=np.float32)
        for k in range(1, self.num_harmonics + 1):
            fourier_basis[:, 2 * (k - 1)] = np.cos(2.0 * math.pi * k * t_fourier)
            fourier_basis[:, 2 * (k - 1) + 1] = np.sin(2.0 * math.pi * k * t_fourier)
        self.register_buffer("basis_season", torch.from_numpy(fourier_basis), persistent=False)  # (H, 2K)
        self.register_buffer("basis_season_t", torch.from_numpy(fourier_basis).t().contiguous(), persistent=False)

        num_trend_coeffs = self.max_degree + 1
        num_season_coeffs = 2 * self.num_harmonics

        # Coefficient Prediction MLPs
        self.trend_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, num_trend_coeffs * channels),
        )

        self.season_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, num_season_coeffs * channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent features into trend, seasonal, and combined forecasts.

        Args:
            x: Latent representation of shape (batch, in_features)

        Returns:
            Tuple of:
            - y_pred: Combined prediction (batch, pred_len, channels)
            - y_trend: Isolated trend prediction (batch, pred_len, channels)
            - y_season: Isolated harmonic prediction (batch, pred_len, channels)
        """
        B = x.shape[0]
        num_p = self.max_degree + 1
        num_k = 2 * self.num_harmonics

        # Predict coefficients: (batch, channels, num_coeffs)
        theta_trend = self.trend_head(x).view(B, self.channels, num_p)
        theta_season = self.season_head(x).view(B, self.channels, num_k)

        # Basis multiplication with pre-transposed buffers
        y_trend = torch.matmul(theta_trend, self.basis_trend_t).permute(0, 2, 1)
        y_season = torch.matmul(theta_season, self.basis_season_t).permute(0, 2, 1)

        y_pred = y_trend + y_season
        return y_pred, y_trend, y_season
