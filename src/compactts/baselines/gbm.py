"""LightGBM gradient boosting baseline for time-series forecasting.

Trains direct models for each horizon step using lag features and rolling statistics.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import lightgbm as lgb


class LightGBMForecaster:
    """Multi-horizon LightGBM regressor with lag and rolling summary features."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "objective": "regression",
            "metric": "l1",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.models: List[lgb.LGBMRegressor] = []
        self.pred_len: int = 0

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract summary and lag features from 2D input (N, context_len).

        Features:
        - Raw lags: last k steps
        - Rolling mean (last 7, 14, 28)
        - Rolling std (last 7, 14)
        - Min, max, slope
        """
        N, L = X.shape
        features = [X]  # Raw lags

        # Rolling statistics
        for window in [min(7, L), min(14, L), min(28, L)]:
            if window > 1:
                r_mean = np.mean(X[:, -window:], axis=1, keepdims=True)
                r_std = np.std(X[:, -window:], axis=1, keepdims=True)
                features.extend([r_mean, r_std])

        # Min, max, trend proxy
        f_min = np.min(X, axis=1, keepdims=True)
        f_max = np.max(X, axis=1, keepdims=True)
        slope = (X[:, -1:] - X[:, :1]) / max(1, L - 1)
        features.extend([f_min, f_max, slope])

        return np.hstack(features).astype(np.float32)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LightGBMForecaster":
        """Fit direct models for each horizon step.

        Args:
            X: Input windows of shape (N, context_len).
            Y: Target windows of shape (N, pred_len).
        """
        if X.ndim == 3:
            X = X[:, :, 0]
        if Y.ndim == 3:
            Y = Y[:, :, 0]

        N, self.pred_len = Y.shape
        feat_matrix = self._extract_features(X)

        self.models = []
        for h in range(self.pred_len):
            reg = lgb.LGBMRegressor(**self.params)
            reg.fit(feat_matrix, Y[:, h])
            self.models.append(reg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pred_len future steps."""
        if X.ndim == 3:
            X = X[:, :, 0]

        feat_matrix = self._extract_features(X)
        preds = []
        for reg in self.models:
            pred_h = reg.predict(feat_matrix)
            preds.append(pred_h)

        # (N, pred_len)
        return np.column_stack(preds).astype(np.float32)
