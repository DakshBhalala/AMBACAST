"""Loss functions for point and probabilistic time-series forecasting.

Supports:
- Huber (Smooth L1) loss & MSE loss for point predictions
- Pinball loss (Quantile Loss) for probabilistic quantile forecasts
- Gaussian Negative Log Likelihood (NLL) for distribution heads
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PinballLoss(nn.Module):
    """Pinball / Quantile loss for multi-quantile forecasting."""

    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute average pinball loss across all quantiles.

        Args:
            y_pred: Predictions of shape (batch, pred_len, channels, num_quantiles)
                    or (batch, pred_len, num_quantiles).
            y_true: Ground truth of shape (batch, pred_len, channels) or (batch, pred_len).
        """
        # Align target shape with predictions
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(-1)

        # Broadcast error: (batch, pred_len, channels, num_quantiles)
        error = y_true - y_pred

        # Quantile loss: max(q * error, (q - 1) * error)
        q_buf = self.quantiles.to(error.device)
        q = q_buf.view(1, 1, 1, -1) if y_pred.dim() == 4 else q_buf.view(1, 1, -1)
        loss = torch.max(q * error, (q - 1.0) * error)
        return torch.mean(loss)


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mean: torch.Tensor, std: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian NLL loss."""
        var = torch.clamp(std ** 2, min=self.eps)
        loss = 0.5 * torch.log(var) + 0.5 * ((y_true - mean) ** 2) / var
        return torch.mean(loss)


def get_loss_function(head_type: str, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)) -> nn.Module:
    """Return appropriate loss function for head type."""
    if head_type == "point":
        return nn.HuberLoss(delta=1.0)
    elif head_type == "quantile":
        return PinballLoss(quantiles=quantiles)
    elif head_type == "distribution":
        return GaussianNLLLoss()
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
