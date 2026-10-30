"""Evaluation suite: metrics, zero-leakage checker, and benchmark runner."""

from compactts.evaluation.metrics import (
    calculate_metrics,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_scaled_error,
    continuous_ranked_probability_score,
    weighted_quantile_loss,
)
from compactts.evaluation.leakage_checker import LeakageChecker
from compactts.evaluation.benchmark import SmokeBenchmarkSuite

__all__ = [
    "calculate_metrics",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_scaled_error",
    "continuous_ranked_probability_score",
    "weighted_quantile_loss",
    "LeakageChecker",
    "SmokeBenchmarkSuite",
]
