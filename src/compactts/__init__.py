"""CompactTS: State-of-the-Art Compact Time-Series Forecasting Library.

Designed for high accuracy, extreme parameter efficiency, and fast inference on consumer hardware and CPUs.
"""

__version__ = "0.2.0"
__author__ = "Daksh Bhalala"

from compactts.models.compact_ts import CompactTS, CompactTSConfig
from compactts.models.compact_ts_pro import CompactTSPro, CompactTSProConfig
from compactts.models.student_t import StudentTHead, StudentTLoss
from compactts.models.functional_basis import FunctionalBasisDecoder
from compactts.models.state_space import DiagonalStateSpaceBlock
from compactts.agent import CompactTSAgent, SeriesProfiler, OpenMetaRouter
from compactts.data.dataset import TimeSeriesDataset, create_sliding_windows
from compactts.data.transforms import RevIN
from compactts.evaluation.metrics import (
    calculate_metrics,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_scaled_error,
    continuous_ranked_probability_score,
    weighted_quantile_loss,
)

__all__ = [
    "CompactTS",
    "CompactTSConfig",
    "CompactTSPro",
    "CompactTSProConfig",
    "CompactTSAgent",
    "SeriesProfiler",
    "OpenMetaRouter",
    "StudentTHead",
    "StudentTLoss",
    "FunctionalBasisDecoder",
    "DiagonalStateSpaceBlock",
    "TimeSeriesDataset",
    "create_sliding_windows",
    "RevIN",
    "calculate_metrics",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_absolute_scaled_error",
    "continuous_ranked_probability_score",
    "weighted_quantile_loss",
    "__version__",
]
