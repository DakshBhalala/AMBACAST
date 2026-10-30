"""Tests for synthetic time series generation."""

import numpy as np
import pytest
from compactts.data.synthetic import SyntheticTimeSeriesGenerator


def test_synthetic_univariate_length_and_types():
    gen = SyntheticTimeSeriesGenerator(seed=42)
    s = gen.generate_univariate(length=500, trend_type="linear", seasonal_periods=(24,))
    assert len(s) == 500
    assert s.dtype == np.float32
    assert not np.any(np.isnan(s))


def test_synthetic_trends():
    gen = SyntheticTimeSeriesGenerator(seed=42)
    s_lin = gen.generate_univariate(length=300, trend_type="linear", trend_slope=0.05, noise_level=0.01)
    s_quad = gen.generate_univariate(length=300, trend_type="quadratic", trend_slope=0.05, noise_level=0.01)
    s_log = gen.generate_univariate(length=300, trend_type="logistic", noise_level=0.01)

    assert s_lin[-1] > s_lin[0]
    assert len(s_quad) == 300
    assert len(s_log) == 300


def test_synthetic_missing_values():
    gen = SyntheticTimeSeriesGenerator(seed=42)
    s = gen.generate_univariate(length=400, missing_rate=0.1)
    nan_count = np.sum(np.isnan(s))
    assert nan_count > 0
    assert nan_count < 400


def test_synthetic_multivariate():
    gen = SyntheticTimeSeriesGenerator(seed=42)
    mv = gen.generate_multivariate(length=500, num_features=4, correlation_strength=0.7)
    assert mv.shape == (500, 4)
    assert mv.dtype == np.float32
    assert not np.any(np.isnan(mv))
