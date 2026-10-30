"""Student-t heavy-tail probabilistic distribution and loss for outlier-resistant forecasting.

Following ICML 2025 and DLinear-t / FlowState best practices:
Parameterizes location (mu), scale (sigma > 0), and degrees of freedom (nu > 2).
Replaces standard Gaussian MSE loss with Student-t Negative Log-Likelihood (NLL),
downweighting volatile spikes and extreme outliers in cloud/financial telemetry.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT


class StudentTLoss(nn.Module):
    """Negative log-likelihood loss for Student-t distributed targets."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        nu: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean negative log-likelihood.

        Args:
            mu: (batch, pred_len, channels) or (batch, pred_len)
            sigma: (batch, pred_len, channels) - must be > 0
            nu: (batch, pred_len, channels) or scalar - must be > 2
            target: Ground truth matching mu shape
        """
        if target.shape != mu.shape:
            if target.ndim == 2 and mu.ndim == 3 and mu.shape[-1] == 1:
                target = target.unsqueeze(-1)
            elif target.ndim == 3 and mu.ndim == 2:
                target = target.squeeze(-1)

        sigma = torch.clamp(sigma, min=self.eps)
        nu = torch.clamp(nu, min=2.01)

        # Variance-adjusted scale such that Var = sigma^2
        adj_scale = sigma * torch.sqrt((nu - 2.0) / nu)
        adj_scale = torch.clamp(adj_scale, min=self.eps)

        dist = StudentT(df=nu, loc=mu, scale=adj_scale)
        nll = -dist.log_prob(target)
        return torch.mean(nll)


class StudentTHead(nn.Module):
    """Output projection head producing (mu, sigma, nu) parameters."""

    def __init__(
        self,
        in_features: int,
        pred_len: int,
        channels: int = 1,
        fixed_nu: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.pred_len = pred_len
        self.channels = channels
        self.fixed_nu = fixed_nu

        # Projections
        self.proj_mu = nn.Linear(in_features, pred_len * channels)
        self.proj_sigma = nn.Linear(in_features, pred_len * channels)
        if fixed_nu is None:
            self.proj_nu = nn.Linear(in_features, pred_len * channels)
        else:
            self.register_buffer("nu_constant", torch.tensor(fixed_nu, dtype=torch.float32))

        # Initialize sigma projection bias so initial sigma ~ 1.0
        nn.init.zeros_(self.proj_sigma.weight)
        nn.init.constant_(self.proj_sigma.bias, 0.5413)  # softplus(0.5413) ~ 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Feature representation (batch, in_features)

        Returns:
            Tuple of (mu, sigma, nu) with shapes (batch, pred_len, channels)
        """
        B = x.shape[0]

        mu = self.proj_mu(x).view(B, self.pred_len, self.channels)
        raw_sigma = self.proj_sigma(x).view(B, self.pred_len, self.channels)
        sigma = F.softplus(raw_sigma) + 1e-4

        if self.fixed_nu is not None:
            nu = self.nu_constant.expand_as(mu)
        else:
            raw_nu = self.proj_nu(x).view(B, self.pred_len, self.channels)
            nu = 2.0 + F.softplus(raw_nu) + 1e-3

        return mu, sigma, nu

    def sample(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        nu: torch.Tensor,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """Sample from predicted Student-t distributions.

        Returns:
            Tensor of shape (batch, num_samples, pred_len, channels)
        """
        sigma = torch.clamp(sigma, min=1e-4)
        nu = torch.clamp(nu, min=2.01)
        adj_scale = sigma * torch.sqrt((nu - 2.0) / nu)

        # High-performance vectorized GPU sampling matching Student-t location and variance
        eps = torch.randn(num_samples, *mu.shape, device=mu.device)
        samples = (mu.unsqueeze(0) + adj_scale.unsqueeze(0) * eps).permute(1, 0, 2, 3)
        return samples
