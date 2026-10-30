"""Unit tests for CompactTS-Agent scale-invariant mathematical routing components."""

import numpy as np
import pytest
from compactts.agent.feature_extractor import TimeSeriesFeatureExtractor
from compactts.agent.meta_router import OpenMetaRouter
from compactts.agent.post_processor import CalibratedPostProcessor


def test_feature_extractor_scale_invariance():
    """Verify that feature extraction is mathematically scale-invariant."""
    fe = TimeSeriesFeatureExtractor(freq="H")
    rng = np.random.default_rng(42)
    y1 = np.sin(np.linspace(0, 10 * np.pi, 100)) + 2.0 + rng.normal(0, 0.05, 100)
    y2 = y1 * 5000.0  # Scale by 5000x

    f1 = fe.extract(y1, horizon=24)
    f2 = fe.extract(y2, horizon=24)

    for k in ["kurtosis", "mean_shift", "spectral_entropy", "volatility_clustering"]:
        diff = abs(f1[k] - f2[k])
        assert diff < 1e-4, f"Feature {k} violated scale invariance: {f1[k]} vs {f2[k]}"


def test_open_meta_router_spectral_regime():
    """Verify routing on pure spectral entropy."""
    router = OpenMetaRouter()
    # Sharp harmonic / diurnal oscillation (low spectral entropy)
    expert = router.route_profile({"spectral_entropy": 0.15, "mean_shift": 0.05})
    assert expert == "CastStar"


def test_open_meta_router_zero_drift_telemetry():
    """Verify routing on stationary baseline telemetry."""
    router = OpenMetaRouter()
    expert = router.route_profile({"spectral_entropy": 0.50, "mean_shift": 0.01})
    assert expert == "chronos_bolt_base"


def test_open_meta_router_volatility_clustering():
    """Verify routing on ARCH volatility clustering."""
    router = OpenMetaRouter()
    # High volatility clustering -> Autoregressive transformer
    expert_high_vol = router.route_profile({
        "spectral_entropy": 0.50,
        "mean_shift": 0.05,
        "volatility_clustering": 0.15,
    })
    assert expert_high_vol == "Timestral-209M"

    # Low volatility clustering -> CompactTS native backbone
    expert_low_vol = router.route_profile({
        "spectral_entropy": 0.50,
        "mean_shift": 0.05,
        "volatility_clustering": 0.01,
    })
    assert expert_low_vol == "CompactTS-Agent"


def test_calibrated_post_processor_monotonicity():
    processor = CalibratedPostProcessor(timestral_quantile_weight=0.0)
    q = np.array([
        [0.9, 0.8],
        [0.1, 0.2],
        [0.5, 0.4],
        [0.3, 0.3],
        [0.7, 0.6],
        [0.2, 0.1],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
    ])
    processed = processor.process_quantiles(q, non_negative=True)
    assert processed.shape == (9, 2)
    for t in range(2):
        for i in range(8):
            assert processed[i, t] <= processed[i + 1, t] + 1e-6
