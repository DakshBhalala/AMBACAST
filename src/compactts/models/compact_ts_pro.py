"""CompactTS-Pro: State-of-the-Art Compact Time-Series Forecasting Architecture.

Combines:
1. Reversible Instance Normalization (RevIN) with learned affine scaling.
2. Continuous-Time Diagonal State-Space Model (S5 / S4D) layer for sampling-rate invariance.
3. Multi-kernel dilated temporal convolutions with Gated Linear Unit (GLU) temporal mixing.
4. Orthogonal Functional Basis Decoder (Chebyshev polynomial trend + Fourier harmonics).
5. Heavy-tail Student-t Likelihood Head (location mu, scale sigma, degrees-of-freedom nu).
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from compactts.data.transforms import RevIN
from compactts.models.components import MultiKernelTemporalConv, GatedTemporalMLP, MovingAverage
from compactts.models.state_space import DiagonalStateSpaceBlock
from compactts.models.functional_basis import FunctionalBasisDecoder
from compactts.models.student_t import StudentTHead, StudentTLoss


@dataclass
class CompactTSProConfig:
    """Hyperparameter configuration for CompactTS-Pro."""
    context_len: int = 96
    pred_len: int = 24
    num_features: int = 1
    d_model: int = 64
    d_state: int = 16
    kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    dilation_rates: Tuple[int, ...] = (1, 2, 4)
    num_glu_layers: int = 2
    max_polynomial_degree: int = 4
    num_harmonics: int = 8
    dropout: float = 0.1
    use_revin: bool = True
    affine_revin: bool = True
    fixed_nu: Optional[float] = None
    num_samples: int = 100

    @classmethod
    def from_tier(cls, tier: str = "pro", context_len: int = 96, pred_len: int = 24, num_features: int = 1) -> "CompactTSProConfig":
        """Factory configuration presets."""
        presets = {
            "pro_compact": {
                "d_model": 48,
                "d_state": 12,
                "num_glu_layers": 2,
                "num_harmonics": 6,
            },
            "pro": {
                "d_model": 64,
                "d_state": 16,
                "num_glu_layers": 2,
                "num_harmonics": 8,
            },
            "pro_large": {
                "d_model": 96,
                "d_state": 24,
                "num_glu_layers": 3,
                "num_harmonics": 12,
            },
        }
        cfg_dict = presets.get(tier, presets["pro"]).copy()
        cfg_dict.update({
            "context_len": context_len,
            "pred_len": pred_len,
            "num_features": num_features,
        })
        return cls(**cfg_dict)


class CompactTSPro(nn.Module):
    """CompactTS-Pro Architecture targeting Top-5 GIFT-Eval Leaderboard performance."""

    def __init__(self, config: Optional[CompactTSProConfig] = None) -> None:
        super().__init__()
        self.config = config or CompactTSProConfig()
        cfg = self.config

        # 1. Reversible Instance Normalization
        if cfg.use_revin:
            self.revin = RevIN(num_features=cfg.num_features, affine=cfg.affine_revin)
        else:
            self.revin = None

        # 2. Input Embedding
        self.input_embed = nn.Linear(cfg.num_features, cfg.d_model)

        # 3. Continuous-Time Diagonal State-Space Layer
        self.ssm_block = DiagonalStateSpaceBlock(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
        )

        # 4. Multi-Kernel Temporal Convolutions + GLU Temporal Mixing
        self.temporal_conv = MultiKernelTemporalConv(
            channels=cfg.d_model,
            kernel_sizes=cfg.kernel_sizes,
            dilations=cfg.dilation_rates,
        )

        self.glu_layers = nn.ModuleList([
            GatedTemporalMLP(seq_len=cfg.context_len, expansion=2, dropout=cfg.dropout)
            for _ in range(cfg.num_glu_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

        # 5. Dual Decoding:
        # 5a. Orthogonal Functional Basis Decoder (Chebyshev polynomial + Fourier harmonic bases)
        self.latent_proj = nn.Linear(cfg.context_len * cfg.d_model, cfg.d_model * 2)
        self.basis_decoder = FunctionalBasisDecoder(
            in_features=cfg.d_model * 2,
            pred_len=cfg.pred_len,
            channels=cfg.num_features,
            max_polynomial_degree=cfg.max_polynomial_degree,
            num_harmonics=cfg.num_harmonics,
        )

        # 5b. Residual Point Adjustment
        self.residual_head = nn.Linear(cfg.d_model * 2, cfg.pred_len * cfg.num_features)

        # 6. Heavy-Tail Student-t Uncertainty Head (predicts mu, sigma, nu)
        self.student_t_head = StudentTHead(
            in_features=cfg.d_model * 2,
            pred_len=cfg.pred_len,
            channels=cfg.num_features,
            fixed_nu=cfg.fixed_nu,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, context_len, num_features) or (batch, context_len)

        Returns:
            Tuple of:
            - mu: Point forecast / Student-t location (batch, pred_len, num_features)
            - sigma: Predictive scale / uncertainty (batch, pred_len, num_features)
            - nu: Degrees of freedom / tail heaviness (batch, pred_len, num_features)
        """
        was_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            was_2d = True

        B, L, C = x.shape

        # Step 1: RevIN Normalization
        if self.revin is not None:
            x_norm = self.revin(x, mode="norm")
        else:
            x_norm = x

        # Step 2: Linear projection to hidden dimension
        h = self.input_embed(x_norm)  # (B, L, d_model)

        # Step 3: Continuous-Time State-Space Processing
        h = self.ssm_block(h)  # (B, L, d_model)

        # Step 4: Multi-Kernel Temporal Convolutions
        h_conv = self.temporal_conv(h)  # (B, L, d_model)

        # Step 5: Stacked GLU Temporal Mixing (channels-first: (B, d_model, L))
        h_glu = h_conv.permute(0, 2, 1)
        for glu in self.glu_layers:
            h_glu = glu(h_glu)

        # Back to (B, L, d_model)
        h_feat = self.norm(h_glu.permute(0, 2, 1))

        # Flatten representation across time
        h_flat = h_feat.reshape(B, -1)
        latent = F.silu(self.latent_proj(h_flat))  # (B, d_model * 2)

        # Step 6: Functional Basis Projection (Smooth Trend + Fourier Harmonics)
        basis_pred, _, _ = self.basis_decoder(latent)  # (B, H, C)
        res_pred = self.residual_head(latent).view(B, self.config.pred_len, C)
        raw_mu = basis_pred + 0.1 * res_pred

        # Step 7: Student-t Parameter Estimation
        _, sigma, nu = self.student_t_head(latent)

        # Step 8: Inverse Normalization for mu and sigma
        if self.revin is not None:
            mu = self.revin(raw_mu, mode="denorm")
            # Scale sigma by revin standard deviation
            if hasattr(self.revin, "stdev") and self.revin.stdev is not None:
                sigma = sigma * self.revin.stdev
        else:
            mu = raw_mu

        if was_2d and C == 1:
            mu = mu.squeeze(-1)
            sigma = sigma.squeeze(-1)
            nu = nu.squeeze(-1)

        return mu, sigma, nu

    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Generate Monte Carlo samples from predictive distribution."""
        mu, sigma, nu = self.forward(x)
        if mu.ndim == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            nu = nu.unsqueeze(-1)
        samples = self.student_t_head.sample(mu, sigma, nu, num_samples=num_samples)
        return samples  # (batch, num_samples, pred_len, channels)
