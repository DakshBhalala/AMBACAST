"""Architectural components for CompactTS.

Includes:
- Multi-scale moving-average decomposition
- Multi-kernel depthwise-separable temporal convolutions
- Gated residual temporal mixing (GLU)
- Lightweight patching projection
- Optional cross-channel mixing
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """Moving average filter for seasonal-trend decomposition."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average along time dimension.

        Args:
            x: Tensor of shape (batch, seq_len, channels).

        Returns:
            Smoothed tensor of shape (batch, seq_len, channels).
        """
        # Padding on both sides to keep the same sequence length
        front_pad = (self.kernel_size - 1) // 2
        end_pad = (self.kernel_size - 1) - front_pad

        # Permute to (batch, channels, seq_len) for 1D pooling
        x_perm = x.permute(0, 2, 1)

        # Replicate padding at edges to avoid distortion
        front = x_perm[:, :, :1].repeat(1, 1, front_pad)
        end = x_perm[:, :, -1:].repeat(1, 1, end_pad)
        x_padded = torch.cat([front, x_perm, end], dim=2)

        out = self.avg(x_padded)
        return out.permute(0, 2, 1)


class MultiScaleDecomposition(nn.Module):
    """Multi-scale moving-average decomposition block.

    Separates sequence into macro-trend, seasonal/harmonic, and high-frequency residual.
    """

    def __init__(self, kernel_sizes: Tuple[int, ...] = (13, 25)) -> None:
        super().__init__()
        self.ma_layers = nn.ModuleList([MovingAverage(k) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose input into trend and seasonal/residual parts.

        Args:
            x: Tensor of shape (batch, seq_len, channels).

        Returns:
            Tuple of (trend, seasonal_residual).
        """
        trends = [ma(x) for ma in self.ma_layers]
        combined_trend = torch.stack(trends, dim=0).mean(dim=0)
        seasonal_residual = x - combined_trend
        return combined_trend, seasonal_residual


class MultiKernelTemporalConv(nn.Module):
    """Depthwise-separable multi-kernel 1D convolutions with dilations.

    Captures multi-scale periodicities and transient shocks with minimal parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 15),
        dilations: Tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.conv_branches = nn.ModuleList()

        for k, d in zip(kernel_sizes, dilations):
            padding = (k - 1) * d // 2
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    dilation=d,
                    padding=padding,
                    groups=channels,  # Depthwise
                    bias=False,
                ),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,  # Pointwise
                    bias=True,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            self.conv_branches.append(branch)

        self.fuse = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-kernel depthwise convolutions.

        Args:
            x: Tensor of shape (batch, seq_len, channels).

        Returns:
            Tensor of shape (batch, seq_len, channels).
        """
        # (batch, channels, seq_len)
        x_perm = x.permute(0, 2, 1)
        branch_outs = [b(x_perm) for b in self.conv_branches]
        cat = torch.cat(branch_outs, dim=1)
        fused = self.fuse(cat)
        # Residual connection
        out = (fused + x_perm).permute(0, 2, 1)
        return out


class GatedTemporalMLP(nn.Module):
    """Residual Gated Linear Unit (GLU) temporal mixing block."""

    def __init__(self, seq_len: int, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = seq_len * expansion
        self.fc1 = nn.Linear(seq_len, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix along temporal dimension.

        Args:
            x: Tensor of shape (batch, channels, seq_len).

        Returns:
            Tensor of shape (batch, channels, seq_len).
        """
        residual = x
        x_norm = self.norm(x)
        h = self.fc1(x_norm)
        # Gated activation: split into value and gate
        val, gate = torch.chunk(h, 2, dim=-1)
        h = val * torch.sigmoid(gate)
        h = self.dropout(h)
        out = self.fc2(h)
        return residual + out


class CrossChannelMixing(nn.Module):
    """Lightweight 1x1 convolution for cross-channel / multivariate interaction."""

    def __init__(self, num_channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix across channels.

        Args:
            x: (batch, seq_len, channels)
        """
        if x.size(-1) == 1:
            return x  # Single channel, skip

        # (batch, channels, seq_len)
        x_perm = x.permute(0, 2, 1)
        mixed = self.conv(x_perm)
        out = self.norm(mixed + x_perm)
        return out.permute(0, 2, 1)
