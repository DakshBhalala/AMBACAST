"""Tests for authentic GIFT-Eval data loader."""

import pytest
import numpy as np
from compactts.data.gifteval_loader import GiftEvalDataLoader, GIFT_EVAL_CATALOG


def test_catalog_structure():
    """Verify that catalog metadata has all required benchmark fields."""
    assert "ett1_hourly" in GIFT_EVAL_CATALOG
    assert "jena_weather_hourly" in GIFT_EVAL_CATALOG
    assert "us_births_daily" in GIFT_EVAL_CATALOG
    assert "sz_taxi_hourly" in GIFT_EVAL_CATALOG

    for k, v in GIFT_EVAL_CATALOG.items():
        assert "repo_path" in v
        assert "domain" in v
        assert "frequency" in v
        assert "seasonality" in v
        assert "default_context" in v
        assert "default_horizon" in v


def test_loader_load_cached_dataset():
    """Verify loading of authentic dataset yields valid 1D time series."""
    loader = GiftEvalDataLoader()
    # ETT1 is small (0.47MB) and already downloaded to cache
    arr, meta = loader.load_dataset("ett1_hourly", univariate=True)

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert len(arr) == 17420
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))
    assert meta["domain"] == "Energy"
    assert meta["seasonality"] == 24
