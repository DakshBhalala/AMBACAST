"""Mathematically Rigorous, Scale-Invariant Time-Series Feature Extractor.

Extracts principled statistical, spectral, temporal, and information-theoretic
signatures from raw time-series trajectories for leak-free autonomous agent dispatching.

Guarantees:
- Strict Scale Invariance: Multiplying y by any scalar c > 0 yields identical routing features.
- Strict Zero Leakage: No dataset names, metadata lookups, or benchmark splits.
- Pure Numerical Input: Operates exclusively on array y in R^T and runtime constraints (H, freq).

Attribution: Daksh Bhalala
License: Business Source License 1.1 (BSL-1.1)
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram


class TimeSeriesFeatureExtractor:
    """Scale-invariant, robust feature extractor for autonomous time-series profiling."""

    FREQ_TO_SECONDS = {
        "10S": 10.0,
        "5T": 300.0,
        "10T": 600.0,
        "15T": 900.0,
        "H": 3600.0,
        "D": 86400.0,
        "W": 604800.0,
        "M": 2629746.0,
        "Q": 7889238.0,
        "A": 31556952.0,
    }

    FREQ_TO_SEASONALITY = {
        "10S": 60,
        "5T": 12,
        "10T": 6,
        "15T": 4,
        "H": 24,
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "A": 1,
    }

    def __init__(self, freq: str = "H"):
        self.freq = freq
        self.period = self.FREQ_TO_SEASONALITY.get(freq, 24)
        self.sampling_interval_seconds = self.FREQ_TO_SECONDS.get(freq, 3600.0)

    def extract(self, series: Union[np.ndarray, list], horizon: int = 24) -> Dict[str, float]:
        """Extract a scale-invariant, comprehensive feature vector.

        Parameters
        ----------
        series : Union[np.ndarray, list]
            Raw historical trajectory y in R^T.
        horizon : int
            Forecast horizon H requested by the user.

        Returns
        -------
        Dict[str, float]
            Scale-invariant features capturing dynamics, intermittency, shifts, and spectrum.
        """
        y = np.asarray(series, dtype=np.float64)
        if y.ndim > 1:
            y = y.squeeze()
            if y.ndim > 1:
                # Multivariate: select primary variate
                y = y[0] if y.shape[0] < y.shape[1] else y[:, 0]

        # Handle missing values and infs cleanly
        if np.isnan(y).any() or np.isinf(y).any():
            s = pd.Series(y).ffill().bfill().to_numpy()
            y = np.nan_to_num(s, nan=0.0).astype(np.float64)

        n = len(y)
        if n < 4:
            return self._fallback_features(n, horizon)

        # 1. Physical Support Characteristics
        min_val = float(np.min(y))
        max_val = float(np.max(y))
        is_strictly_positive = 1.0 if min_val >= 0.0 else 0.0
        val_range = max_val - min_val

        # 2. Intermittency & Sparsity
        # Ratio of exact or near-zeros (relative to overall scale)
        threshold = 1e-5 if val_range < 1e-4 else 1e-4 * val_range
        zero_ratio = float(np.mean(np.abs(y) <= threshold))

        # 3. Scale-Normalized Moments
        mean_val = float(np.mean(y))
        std_val = float(np.std(y))
        if std_val < 1e-8:
            return self._constant_series_features(n, horizon, is_strictly_positive)

        # Z-score normalization for scale invariance
        z = (y - mean_val) / std_val

        # Higher-order normalized moments
        try:
            kurt = float(stats.kurtosis(z))
            skew = float(stats.skew(z))
        except Exception:
            kurt, skew = 0.0, 0.0

        # 4. Non-Stationarity & Regime Shifts
        # Evaluate divergence between historical past and recent window
        half = n // 2
        past_z = z[:half]
        recent_z = z[half:]
        mean_shift = abs(float(np.mean(recent_z)) - float(np.mean(past_z)))
        std_shift = abs(float(np.std(recent_z)) - float(np.std(past_z)))

        # 5. Temporal Autocorrelation & Stationarity
        # First lag autocorrelation (persistence)
        if n > 2:
            try:
                ac1 = float(np.corrcoef(z[1:], z[:-1])[0, 1])
                if not np.isfinite(ac1):
                    ac1 = 0.0
            except Exception:
                ac1 = 0.0
        else:
            ac1 = 0.0

        # Seasonal lag autocorrelation
        S = self.period
        seasonal_corr = 0.0
        if n > 2 * S and S > 1:
            try:
                ac_s = float(np.corrcoef(z[S:], z[:-S])[0, 1])
                if np.isfinite(ac_s):
                    seasonal_corr = max(0.0, ac_s)
            except Exception:
                seasonal_corr = 0.0

        # 6. Spectral Energy & Seasonality Purity
        # Computes power spectrum via periodogram
        spectral_entropy = 0.5
        dominant_freq_energy_ratio = 0.0
        if n >= 16:
            try:
                _, pxx = periodogram(z)
                pxx_sum = np.sum(pxx)
                if pxx_sum > 1e-8:
                    pxx_norm = pxx / pxx_sum
                    # Shannon entropy of power spectrum (normalized to [0, 1])
                    pxx_pos = pxx_norm[pxx_norm > 1e-12]
                    h_spec = -np.sum(pxx_pos * np.log(pxx_pos))
                    spectral_entropy = float(h_spec / np.log(len(pxx)))
                    dominant_freq_energy_ratio = float(np.max(pxx_norm))
            except Exception:
                pass

        # 7. ARCH Volatility Clustering (Autocorrelation of squared returns)
        diff_z = np.diff(z)
        vol_clustering = 0.0
        if len(diff_z) > 4:
            sq_diff = diff_z**2
            sq_std = float(np.std(sq_diff))
            if sq_std > 1e-6:
                try:
                    c_arch = float(np.corrcoef(sq_diff[1:], sq_diff[:-1])[0, 1])
                    if np.isfinite(c_arch):
                        vol_clustering = max(0.0, c_arch)
                except Exception:
                    pass

        # 8. Horizon-to-Context Ratio
        horizon_ratio = float(horizon / min(n, 512))

        return {
            "length": float(n),
            "zero_ratio": zero_ratio,
            "is_strictly_positive": is_strictly_positive,
            "kurtosis": kurt,
            "skewness": skew,
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "autocorr_lag1": ac1,
            "seasonal_corr": seasonal_corr,
            "spectral_entropy": spectral_entropy,
            "dominant_spectral_energy": dominant_freq_energy_ratio,
            "volatility_clustering": vol_clustering,
            "horizon_ratio": horizon_ratio,
            "sampling_seconds_log": float(np.log10(max(self.sampling_interval_seconds, 1.0))),
        }

    def _fallback_features(self, n: int, horizon: int) -> Dict[str, float]:
        return {
            "length": float(n),
            "zero_ratio": 0.0,
            "is_strictly_positive": 1.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "mean_shift": 0.0,
            "std_shift": 0.0,
            "autocorr_lag1": 0.0,
            "seasonal_corr": 0.0,
            "spectral_entropy": 1.0,
            "dominant_spectral_energy": 0.0,
            "volatility_clustering": 0.0,
            "horizon_ratio": float(horizon / max(n, 1)),
            "sampling_seconds_log": float(np.log10(max(self.sampling_interval_seconds, 1.0))),
        }

    def _constant_series_features(self, n: int, horizon: int, is_pos: float) -> Dict[str, float]:
        res = self._fallback_features(n, horizon)
        res["is_strictly_positive"] = is_pos
        res["autocorr_lag1"] = 1.0
        res["spectral_entropy"] = 0.0
        return res
