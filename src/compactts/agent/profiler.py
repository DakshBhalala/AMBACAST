"""Series Profiler for CompactTS-Agent.

Extracts statistical signatures, spectral characteristics, seasonality,
trend strength, intermittency, and regime shifts from time-series trajectories.

Attribution: Daksh Bhalala
License: Business Source License 1.1 (BSL-1.1)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from gluonts.time_feature import get_seasonality


class SeriesProfiler:
    """Profiles univariate and multivariate time series for autonomous agent arbitration."""

    def __init__(self, freq: str = "H"):
        self.freq = freq
        try:
            self.base_seasonality = max(get_seasonality(freq), 1)
        except Exception:
            freq_map = {
                "10S": 60, "5T": 12, "10T": 6, "15T": 4,
                "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "A": 1
            }
            self.base_seasonality = freq_map.get(freq, 24)

    def profile(self, series: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive statistical and structural characteristics.

        Parameters
        ----------
        series : np.ndarray
            1D or 2D array of historical observations.

        Returns
        -------
        Dict[str, Any]
            Dictionary of structural features:
            - length: sequence length
            - mean, std: empirical scale
            - zero_fraction: proportion of near-zero values (intermittency/solar)
            - cv: coefficient of variation (volatility)
            - kurtosis: tail weight indicator
            - seasonal_period: detected dominant lag
            - seasonal_strength: autocorrelation at seasonal lag
            - trend_strength: relative trend variance ratio
            - mean_shift: shift in mean between first and second half (non-stationarity)
            - std_shift: shift in std between first and second half
            - has_missing: boolean indicating gaps or NaNs
        """
        y = np.asarray(series, dtype=np.float32)
        if y.ndim > 1:
            y = y.squeeze()
            if y.ndim > 1:
                # If multivariate, profile primary target channel
                y = y[0] if y.shape[0] < y.shape[1] else y[:, 0]

        has_missing = bool(np.isnan(y).any() or np.isinf(y).any())
        if has_missing:
            s = pd.Series(y).ffill().bfill().to_numpy()
            y = np.nan_to_num(s, nan=0.0).astype(np.float32)

        n = len(y)
        if n < 4:
            return {
                "length": n,
                "mean": float(np.mean(y)) if n > 0 else 0.0,
                "std": float(np.std(y)) if n > 0 else 1.0,
                "zero_fraction": 0.0,
                "cv": 1.0,
                "kurtosis": 0.0,
                "seasonal_period": self.base_seasonality,
                "seasonal_strength": 0.0,
                "trend_strength": 0.0,
                "mean_shift": 0.0,
                "std_shift": 0.0,
                "has_missing": has_missing,
            }

        mean_val = float(np.mean(y))
        std_val = float(np.std(y))
        if std_val < 1e-8:
            std_val = 1.0

        zero_fraction = float(np.mean(np.abs(y) < 1e-5))
        cv = std_val / (abs(mean_val) + 1e-5)

        # Kurtosis for heavy-tail detection
        try:
            kurt = float(stats.kurtosis(y))
        except Exception:
            kurt = 0.0

        # Non-stationarity & regime shift metrics (split context in half)
        half = n // 2
        m1, m2 = float(np.mean(y[:half])), float(np.mean(y[half:]))
        s1, s2 = float(np.std(y[:half])), float(np.std(y[half:]))
        mean_shift = abs(m1 - m2) / (std_val + 1e-5)
        std_shift = abs(s1 - s2) / (std_val + 1e-5)

        # Seasonal autocorrelation
        S = self.base_seasonality
        s_corr = 0.0
        if n > 2 * S and S > 1:
            try:
                c = np.corrcoef(y[S:], y[:-S])[0, 1]
                if np.isfinite(c):
                    s_corr = max(0.0, float(c))
            except Exception:
                s_corr = 0.0

        # Trend strength: ratio of diff variance to raw variance
        diff_var = float(np.var(np.diff(y)))
        raw_var = float(np.var(y))
        trend_ratio = max(0.0, 1.0 - (diff_var / (raw_var + 1e-8)))

        return {
            "length": n,
            "mean": mean_val,
            "std": std_val,
            "zero_fraction": zero_fraction,
            "cv": cv,
            "kurtosis": kurt,
            "seasonal_period": S,
            "seasonal_strength": s_corr,
            "trend_strength": trend_ratio,
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "has_missing": has_missing,
        }
