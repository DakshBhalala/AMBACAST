"""CompactTS: State-of-the-Art Compact Time-Series Forecasting Model.

Architecture:
1. Reversible Instance Normalization (RevIN) for distribution stability.
2. Multi-Scale Hierarchical Moving-Average Decomposition (Trend vs. Seasonal/Residual).
3. Dual-Branch Processing:
   - Trend Branch: Linear / smooth polynomial projection.
   - Seasonal/Harmonic Branch: Multi-kernel dilated depthwise convolutions + Residual Gated Linear Unit (GLU) temporal mixing.
4. Channel Independence (default for universal cross-dataset transfer) or Optional Cross-Channel Mixing.
5. Flexible Multi-Head: Point, Quantile, or Gaussian Distribution.
"""

from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from compactts.data.transforms import RevIN
from compactts.models.components import (
    MultiScaleDecomposition,
    MultiKernelTemporalConv,
    GatedTemporalMLP,
    CrossChannelMixing,
)
from compactts.models.heads import PointHead, QuantileHead, DistributionHead


@dataclass
class CompactTSConfig:
    """Configuration class for CompactTS models."""

    context_len: int = 96
    pred_len: int = 24
    num_features: int = 1
    d_model: int = 64
    num_layers: int = 2
    expansion: int = 2
    decomp_kernel_sizes: Tuple[int, ...] = (13, 25)
    conv_kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    conv_dilations: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    use_revin: bool = True
    affine_revin: bool = True
    use_cross_channel: bool = False
    head_type: str = "point"  # 'point', 'quantile', or 'distribution'
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    hidden_dim: Optional[int] = 128

    @classmethod
    def from_tier(
        cls,
        tier: str = "small",
        context_len: int = 96,
        pred_len: int = 24,
        num_features: int = 1,
        head_type: str = "point",
        use_cross_channel: bool = False,
    ) -> "CompactTSConfig":
        """Instantiate configuration for designated parameter budget tiers.

        Tiers:
            - tiny: ~100k parameters
            - small: ~500k-1M parameters
            - medium: ~3M-5M parameters
            - large: ~10M parameters
        """
        tier = tier.lower()
        if tier == "tiny":
            return cls(
                context_len=context_len,
                pred_len=pred_len,
                num_features=num_features,
                d_model=32,
                num_layers=1,
                expansion=2,
                hidden_dim=64,
                head_type=head_type,
                use_cross_channel=use_cross_channel,
            )
        elif tier == "small":
            return cls(
                context_len=context_len,
                pred_len=pred_len,
                num_features=num_features,
                d_model=64,
                num_layers=2,
                expansion=2,
                hidden_dim=128,
                head_type=head_type,
                use_cross_channel=use_cross_channel,
            )
        elif tier == "medium":
            return cls(
                context_len=context_len,
                pred_len=pred_len,
                num_features=num_features,
                d_model=128,
                num_layers=3,
                expansion=2,
                hidden_dim=256,
                head_type=head_type,
                use_cross_channel=use_cross_channel,
            )
        elif tier == "large":
            return cls(
                context_len=context_len,
                pred_len=pred_len,
                num_features=num_features,
                d_model=256,
                num_layers=4,
                expansion=2,
                hidden_dim=512,
                head_type=head_type,
                use_cross_channel=use_cross_channel,
            )
        else:
            raise ValueError(f"Unknown tier: {tier}. Choose from 'tiny', 'small', 'medium', 'large'")


class CompactTS(nn.Module):
    """Compact Time-Series Forecasting Model."""

    def __init__(self, config: CompactTSConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Normalization
        if config.use_revin:
            self.revin = RevIN(
                num_features=config.num_features,
                affine=config.affine_revin,
                eps=1e-5,
            )
        else:
            self.revin = None

        # 2. Multi-Scale Decomposition
        self.decomposition = MultiScaleDecomposition(kernel_sizes=config.decomp_kernel_sizes)

        # 3. Trend Projection Branch (Direct Linear from context to horizon)
        self.trend_projector = nn.Linear(config.context_len, config.pred_len)

        # 4. Seasonal & Harmonic Processing Branch
        # Temporal Embedding: projects from raw scalar per step to d_model
        self.seasonal_embed = nn.Linear(1, config.d_model)

        # Multi-scale Convolutional filtering
        self.conv_block = MultiKernelTemporalConv(
            channels=config.d_model,
            kernel_sizes=config.conv_kernel_sizes,
            dilations=config.conv_dilations,
        )

        # Gated Temporal Mixing Stack
        self.temporal_layers = nn.ModuleList([
            GatedTemporalMLP(
                seq_len=config.context_len,
                expansion=config.expansion,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Dimension reduction back to 1 feature
        self.seasonal_reduce = nn.Linear(config.d_model, 1)

        # 5. Cross-Channel Interaction (Optional)
        if config.use_cross_channel and config.num_features > 1:
            self.cross_channel = CrossChannelMixing(num_channels=config.num_features)
        else:
            self.cross_channel = None

        # 6. Forecasting Head
        if config.head_type == "point":
            self.head = PointHead(
                in_features=config.context_len,
                pred_len=config.pred_len,
                hidden_dim=config.hidden_dim,
            )
        elif config.head_type == "quantile":
            self.head = QuantileHead(
                in_features=config.context_len,
                pred_len=config.pred_len,
                quantiles=config.quantiles,
                hidden_dim=config.hidden_dim,
            )
        elif config.head_type == "distribution":
            self.head = DistributionHead(
                in_features=config.context_len,
                pred_len=config.pred_len,
            )
        else:
            raise ValueError(f"Unknown head_type: {config.head_type}")

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, context_len, num_features) or (batch, context_len).

        Returns:
            - If head_type == 'point': Tensor of shape (batch, pred_len, num_features)
            - If head_type == 'quantile': Tensor of shape (batch, pred_len, num_features, num_quantiles)
            - If head_type == 'distribution': Tuple of (mean, std) each (batch, pred_len, num_features)
        """
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            was_2d = True

        batch_size, seq_len, num_channels = x.shape

        # Step 1: Normalization
        if self.revin is not None:
            x_norm = self.revin(x, mode="norm")
        else:
            x_norm = x

        # Step 2: Cross-Channel Interaction (if enabled)
        if self.cross_channel is not None:
            x_norm = self.cross_channel(x_norm)

        # Step 3: Multi-Scale Decomposition
        trend_in, seasonal_in = self.decomposition(x_norm)

        # Step 4: Trend Branch (forecasts trend directly)
        # Permute to (batch, channels, context_len) for Linear(context_len -> pred_len)
        trend_perm = trend_in.permute(0, 2, 1)
        trend_forecast = self.trend_projector(trend_perm).permute(0, 2, 1)  # (batch, pred_len, channels)

        # Step 5: Seasonal & Harmonic Branch
        # Expand each channel independently: (batch * channels, context_len, 1)
        seasonal_flat = seasonal_in.permute(0, 2, 1).contiguous().view(batch_size * num_channels, seq_len, 1)
        s_emb = self.seasonal_embed(seasonal_flat)  # (B*C, context_len, d_model)

        # Multi-scale convolution
        s_conv = self.conv_block(s_emb)  # (B*C, context_len, d_model)

        # Temporal mixing: permute to (B*C, d_model, context_len)
        s_temp = s_conv.permute(0, 2, 1)
        for layer in self.temporal_layers:
            s_temp = layer(s_temp)

        # Permute back: (B*C, context_len, d_model)
        s_temp = s_temp.permute(0, 2, 1)
        s_red = self.seasonal_reduce(s_temp).squeeze(-1)  # (B*C, context_len)

        # Reshape to (batch, channels, context_len)
        s_features = s_red.view(batch_size, num_channels, seq_len)

        # Step 6: Prediction Head
        if self.config.head_type == "point":
            seasonal_forecast = self.head(s_features)  # (batch, pred_len, channels)
            total_forecast = trend_forecast + seasonal_forecast
            if self.revin is not None:
                total_forecast = self.revin(total_forecast, mode="denorm")
            if was_2d:
                total_forecast = total_forecast.squeeze(-1)
            return total_forecast

        elif self.config.head_type == "quantile":
            # (batch, pred_len, channels, num_quantiles)
            seasonal_forecast = self.head(s_features)
            # Add trend to all quantiles
            trend_expanded = trend_forecast.unsqueeze(-1)
            total_forecast = trend_expanded + seasonal_forecast

            # RevIN denormalize per quantile
            if self.revin is not None:
                q_outs = []
                for q_idx in range(self.config.quantiles.__len__()):
                    q_slice = total_forecast[..., q_idx]
                    q_denorm = self.revin(q_slice, mode="denorm")
                    q_outs.append(q_denorm)
                total_forecast = torch.stack(q_outs, dim=-1)

            if was_2d:
                total_forecast = total_forecast.squeeze(2)
            return total_forecast

        elif self.config.head_type == "distribution":
            # (batch, pred_len, channels)
            mean_s, std_s = self.head(s_features)
            mean = trend_forecast + mean_s
            std = std_s

            if self.revin is not None:
                mean = self.revin(mean, mode="denorm")
                # Scale std by revin stdev
                if self.revin.stdev is not None:
                    std = std * self.revin.stdev

            if was_2d:
                mean = mean.squeeze(-1)
                std = std.squeeze(-1)
            return mean, std
