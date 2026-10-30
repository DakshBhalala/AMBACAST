"""Tests for CompactTS model configurations, output shapes, gradient flow, and checkpointing."""

import os
import tempfile
import torch
import pytest
from compactts.models.compact_ts import CompactTS, CompactTSConfig


@pytest.mark.parametrize("tier", ["tiny", "small", "medium", "large"])
def test_compact_ts_tiers_shapes(tier: str):
    cfg = CompactTSConfig.from_tier(tier, context_len=48, pred_len=12, num_features=1)
    model = CompactTS(cfg)
    x = torch.randn(4, 48, 1)
    out = model(x)
    assert out.shape == (4, 12, 1)
    assert model.count_parameters() > 1000


def test_compact_ts_multivariate():
    cfg = CompactTSConfig(
        context_len=36,
        pred_len=12,
        num_features=3,
        d_model=32,
        num_layers=1,
        use_cross_channel=True,
    )
    model = CompactTS(cfg)
    x = torch.randn(2, 36, 3)
    out = model(x)
    assert out.shape == (2, 12, 3)


def test_compact_ts_quantile_head():
    cfg = CompactTSConfig(
        context_len=48,
        pred_len=12,
        num_features=1,
        d_model=32,
        num_layers=1,
        head_type="quantile",
        quantiles=(0.1, 0.5, 0.9),
    )
    model = CompactTS(cfg)
    x = torch.randn(4, 48, 1)
    out = model(x)
    assert out.shape == (4, 12, 1, 3)


def test_compact_ts_distribution_head():
    cfg = CompactTSConfig(
        context_len=48,
        pred_len=12,
        num_features=1,
        d_model=32,
        num_layers=1,
        head_type="distribution",
    )
    model = CompactTS(cfg)
    x = torch.randn(4, 48, 1)
    mean, std = model(x)
    assert mean.shape == (4, 12, 1)
    assert std.shape == (4, 12, 1)
    assert torch.all(std > 0)


def test_compact_ts_gradient_flow():
    cfg = CompactTSConfig.from_tier("tiny", context_len=32, pred_len=8)
    model = CompactTS(cfg)
    x = torch.randn(2, 32, 1, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"


def test_compact_ts_checkpoint_saving_reloading():
    cfg = CompactTSConfig.from_tier("tiny", context_len=32, pred_len=8)
    model = CompactTS(cfg).eval()
    x = torch.randn(2, 32, 1)

    with torch.no_grad():
        out1 = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(model.state_dict(), tmp_path)
        model2 = CompactTS(cfg).eval()
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))
        with torch.no_grad():
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
