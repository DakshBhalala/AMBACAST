"""Unit tests for CompactTS-Pro architecture and components."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from compactts.models.student_t import StudentTHead, StudentTLoss
from compactts.models.functional_basis import FunctionalBasisDecoder
from compactts.models.state_space import DiagonalStateSpaceBlock
from compactts.models.compact_ts_pro import CompactTSPro, CompactTSProConfig
from compactts.evaluation.gifteval_predictor import CompactTSPredictor


def test_student_t_loss():
    """Test Student-t NLL loss forward and backward."""
    loss_fn = StudentTLoss()
    B, H, C = 4, 12, 1
    mu = torch.randn(B, H, C, requires_grad=True)
    sigma = torch.ones(B, H, C, requires_grad=True)
    nu = torch.full((B, H, C), 4.0, requires_grad=True)
    target = torch.randn(B, H, C)

    loss = loss_fn(mu, sigma, nu, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert mu.grad is not None
    assert sigma.grad is not None
    assert nu.grad is not None


def test_student_t_loss_2d_broadcast():
    """Test Student-t loss handles 2D targets seamlessly."""
    loss_fn = StudentTLoss()
    B, H = 4, 12
    mu = torch.randn(B, H, 1)
    sigma = torch.ones(B, H, 1)
    nu = torch.full((B, H, 1), 5.0)
    target = torch.randn(B, H)

    loss = loss_fn(mu, sigma, nu, target)
    assert torch.isfinite(loss)


def test_student_t_head():
    """Test StudentTHead parameter predictions and sampling."""
    head = StudentTHead(in_features=32, pred_len=24, channels=1)
    x = torch.randn(4, 32)
    mu, sigma, nu = head(x)

    assert mu.shape == (4, 24, 1)
    assert sigma.shape == (4, 24, 1)
    assert nu.shape == (4, 24, 1)
    assert torch.all(sigma > 0)
    assert torch.all(nu > 2.0)

    # Test sampling
    samples = head.sample(mu, sigma, nu, num_samples=50)
    assert samples.shape == (4, 50, 24, 1)


def test_student_t_head_fixed_nu():
    """Test StudentTHead with fixed degrees of freedom."""
    head = StudentTHead(in_features=32, pred_len=16, channels=2, fixed_nu=5.0)
    x = torch.randn(2, 32)
    mu, sigma, nu = head(x)
    assert mu.shape == (2, 16, 2)
    assert torch.allclose(nu, torch.tensor(5.0))


def test_functional_basis_decoder():
    """Test Chebyshev and Fourier orthogonal basis decomposition."""
    decoder = FunctionalBasisDecoder(
        in_features=48,
        pred_len=24,
        channels=1,
        max_polynomial_degree=4,
        num_harmonics=6,
    )
    x = torch.randn(4, 48)
    y_pred, y_trend, y_season = decoder(x)

    assert y_pred.shape == (4, 24, 1)
    assert y_trend.shape == (4, 24, 1)
    assert y_season.shape == (4, 24, 1)
    # y_pred must equal trend + season exactly
    assert torch.allclose(y_pred, y_trend + y_season, atol=1e-5)


def test_functional_basis_decoder_multivariate():
    """Test functional basis decoder with multiple channels."""
    decoder = FunctionalBasisDecoder(
        in_features=32,
        pred_len=12,
        channels=3,
        max_polynomial_degree=3,
        num_harmonics=4,
    )
    x = torch.randn(2, 32)
    y_pred, y_trend, y_season = decoder(x)
    assert y_pred.shape == (2, 12, 3)


def test_diagonal_state_space_block():
    """Test continuous-time state-space layer."""
    d_model = 32
    ssm = DiagonalStateSpaceBlock(d_model=d_model, d_state=8)
    x = torch.randn(4, 48, d_model)
    out = ssm(x)

    assert out.shape == (4, 48, d_model)
    assert torch.isfinite(out).all()


def test_diagonal_state_space_causality():
    """Verify that state-space block maintains strict causal convolution."""
    d_model = 16
    ssm = DiagonalStateSpaceBlock(d_model=d_model, d_state=8)
    ssm.eval()

    L = 20
    x1 = torch.randn(1, L, d_model)
    x2 = x1.clone()
    # Modify only the future part (time t=10 onwards)
    x2[:, 10:, :] = torch.randn(1, 10, d_model)

    with torch.no_grad():
        out1 = ssm(x1)
        out2 = ssm(x2)

    # Output up to index 9 must be identical (no future leakage)
    assert torch.allclose(out1[:, :10, :], out2[:, :10, :], atol=1e-4)


@pytest.mark.parametrize("tier", ["pro_compact", "pro", "pro_large"])
def test_compact_ts_pro_tiers(tier):
    """Test all CompactTSPro preset configurations."""
    cfg = CompactTSProConfig.from_tier(tier=tier, context_len=48, pred_len=16, num_features=1)
    model = CompactTSPro(cfg)
    x = torch.randn(2, 48, 1)
    mu, sigma, nu = model(x)

    assert mu.shape == (2, 16, 1)
    assert sigma.shape == (2, 16, 1)
    assert nu.shape == (2, 16, 1)
    assert torch.all(sigma > 0)
    assert torch.all(nu > 2.0)


def test_compact_ts_pro_2d_input():
    """Test CompactTSPro handles 2D inputs (B, L)."""
    cfg = CompactTSProConfig(context_len=32, pred_len=8, num_features=1)
    model = CompactTSPro(cfg)
    x = torch.randn(3, 32)
    mu, sigma, nu = model(x)

    assert mu.shape == (3, 8)
    assert sigma.shape == (3, 8)
    assert nu.shape == (3, 8)


def test_compact_ts_pro_gradient_flow():
    """Test full backpropagation through CompactTSPro with StudentTLoss."""
    cfg = CompactTSProConfig(context_len=32, pred_len=12, num_features=1)
    model = CompactTSPro(cfg)
    loss_fn = StudentTLoss()

    x = torch.randn(2, 32, 1)
    y_target = torch.randn(2, 12, 1)

    mu, sigma, nu = model(x)
    loss = loss_fn(mu, sigma, nu, y_target)
    loss.backward()

    # Check gradients on key components
    assert model.input_embed.weight.grad is not None
    assert model.ssm_block.log_dt.grad is not None
    assert model.basis_decoder.trend_head[0].weight.grad is not None
    assert model.student_t_head.proj_sigma.weight.grad is not None


def test_compact_ts_pro_sample():
    """Test Monte Carlo sample generation from CompactTSPro."""
    cfg = CompactTSProConfig(context_len=32, pred_len=12, num_features=1)
    model = CompactTSPro(cfg)
    x = torch.randn(2, 32, 1)

    samples = model.sample(x, num_samples=25)
    assert samples.shape == (2, 25, 12, 1)


def test_compact_ts_pro_parameter_count():
    """Confirm CompactTSPro is strictly compact (<1.5M parameters)."""
    cfg = CompactTSProConfig.from_tier("pro", context_len=96, pred_len=24, num_features=1)
    model = CompactTSPro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 1_500_000, f"Expected < 1.5M params, got {total_params}"


def test_compactts_predictor_with_pro():
    """Test CompactTSPredictor integration with CompactTSPro."""
    from gluonts.dataset.common import ListDataset
    import pandas as pd

    cfg = CompactTSProConfig(context_len=32, pred_len=8, num_features=1)
    model = CompactTSPro(cfg)
    model.eval()

    predictor = CompactTSPredictor(
        model=model,
        prediction_length=8,
        context_length=32,
        batch_size=4,
        num_samples=20,
    )

    # Create dummy GluonTS ListDataset
    series = np.sin(np.linspace(0, 10, 50)).astype(np.float32)
    start_period = pd.Period("2024-01-01", freq="D")
    dataset = ListDataset([{"start": start_period, "target": series}], freq="D")

    forecasts = list(predictor.predict(dataset))
    assert len(forecasts) == 1
    fc = forecasts[0]
    assert fc.samples.shape == (20, 8)
    assert np.all(np.isfinite(fc.mean))
