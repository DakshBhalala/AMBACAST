"""Tests for CompactTS-Agent and SeriesProfiler."""

import numpy as np
import pytest
import pandas as pd

from compactts.agent.profiler import SeriesProfiler
from compactts.agent.agent import CompactTSAgent


def test_series_profiler():
    profiler = SeriesProfiler(freq="H")
    # Generate synthetic periodic signal
    t = np.arange(100)
    series = np.sin(2 * np.pi * t / 24) + 0.1 * np.random.randn(100)

    profile = profiler.profile(series)
    assert "length" in profile
    assert profile["length"] == 100
    assert "seasonal_period" in profile
    assert "seasonal_strength" in profile
    assert profile["seasonal_strength"] > 0.5
    assert "kurtosis" in profile
    assert not profile["has_missing"]


def test_compactts_agent_offline():
    # Test agent without foundation backbone (naive quantile expansion fallback)
    agent = CompactTSAgent(backbone_pipeline=None, prediction_length=12, freq="H", batch_size=4)
    
    t = np.arange(50)
    s1 = np.sin(2 * np.pi * t / 24)
    s2 = np.cos(2 * np.pi * t / 24)
    start_dates = [pd.Period("2024-01-01 00:00", freq="h"), pd.Period("2024-01-01 00:00", freq="h")]
    
    forecasts = agent.forecast_batch([s1, s2], start_dates)
    assert len(forecasts) == 2
    for fc in forecasts:
        # Check shapes: (9, 12)
        assert fc.forecast_array.shape == (9, 12)
        # Check monotonicity across quantiles
        for h in range(12):
            q_vals = fc.forecast_array[:, h]
            assert np.all(np.diff(q_vals) >= -1e-5), "Quantiles must be monotonically non-decreasing"
