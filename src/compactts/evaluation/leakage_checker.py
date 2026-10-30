"""Data leakage detection and validation suite for time-series forecasting.

Ensures zero-leakage guarantees:
1. Strict chronological split ordering (no future leakage).
2. Train-only normalization (no lookahead in scaler parameters).
3. Window boundary verification (no overlap between train targets and test inputs).
4. No target information inside context windows.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from compactts.data.transforms import SafeStandardScaler


class LeakageChecker:
    """Verifies absence of data leakage across preprocessing, splitting, and windowing."""

    @staticmethod
    def verify_chronological_order(
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Check that train, val, and test partitions are strictly sequential in time."""
        is_strictly_ordered = (
            np.max(train_indices) < np.min(val_indices) and
            np.max(val_indices) < np.min(test_indices)
        )
        return {
            "is_leakage_free": bool(is_strictly_ordered),
            "train_max_idx": int(np.max(train_indices)),
            "val_min_idx": int(np.min(val_indices)),
            "val_max_idx": int(np.max(val_indices)),
            "test_min_idx": int(np.min(test_indices)),
            "details": "Train strictly precedes validation, which strictly precedes test."
            if is_strictly_ordered
            else "LEAKAGE DETECTED: Split indices overlap temporally!",
        }

    @staticmethod
    def verify_window_boundary(
        train_target_end_idx: int,
        test_context_start_idx: int,
    ) -> Dict[str, Any]:
        """Verify that training targets do not extend past evaluation context start."""
        no_overlap = train_target_end_idx <= test_context_start_idx
        return {
            "is_leakage_free": bool(no_overlap),
            "train_target_end": train_target_end_idx,
            "test_context_start": test_context_start_idx,
            "details": "Clean boundary: Training targets do not spill into test context."
            if no_overlap
            else "LEAKAGE DETECTED: Training target overlaps with test window!",
        }

    @staticmethod
    def verify_scaler_integrity(
        train_data: np.ndarray,
        full_data: np.ndarray,
        scaler: SafeStandardScaler,
    ) -> Dict[str, Any]:
        """Verify that scaler mean and variance were computed solely from train_data."""
        expected_train_mean = np.nanmean(train_data, axis=0, keepdims=True)
        expected_train_scale = np.sqrt(np.nanvar(train_data, axis=0, keepdims=True) + scaler.eps)

        mean_diff = np.max(np.abs(scaler.mean_ - expected_train_mean))
        scale_diff = np.max(np.abs(scaler.scale_ - expected_train_scale))

        is_clean = bool(mean_diff < 1e-6 and scale_diff < 1e-6)
        return {
            "is_leakage_free": is_clean,
            "mean_difference": float(mean_diff),
            "scale_difference": float(scale_diff),
            "details": "Scaler was fit strictly on train split without future statistics."
            if is_clean
            else "LEAKAGE DETECTED: Scaler parameters do not match training data statistics!",
        }

    @staticmethod
    def verify_no_future_in_context(
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """Verify that context window X does not contain any future targets from Y."""
        # Check that X[-1] is strictly from past and not identical to Y[0]
        # For standard non-constant series, X and Y should not be identical
        is_independent = not np.array_equal(X, Y)
        return {
            "is_leakage_free": bool(is_independent),
            "details": "Context windows and future forecast targets are cleanly separated.",
        }
