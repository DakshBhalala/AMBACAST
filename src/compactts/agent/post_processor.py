"""Calibrated Post-Processor for CompactTS-Agent-v2.

Enforces physical validity, quantile monotonicity, and probabilistic
interval sharpness for state-of-the-art CRPS calibration.

Attribution: Daksh Bhalala
License: Business Source License 1.1 (BSL-1.1)
"""

from typing import Optional
import numpy as np


class CalibratedPostProcessor:
    """Post-processing pipeline for probabilistic forecasts.

    Ensures:
    1. Strict quantile non-crossing: q_0.1 <= q_0.2 <= ... <= q_0.9
    2. Physical support constraints: non-negativity for energy, volume, counts, solar
    3. Probabilistic quantile calibration via dispersion blending
    """

    def __init__(self, timestral_quantile_weight: float = 0.0):
        self.w_tms = timestral_quantile_weight

    def process_quantiles(
        self,
        quantiles: np.ndarray,
        aux_quantiles: Optional[np.ndarray] = None,
        non_negative: bool = False,
    ) -> np.ndarray:
        """Process and calibrate a 9-quantile forecast array.

        Parameters
        ----------
        quantiles : np.ndarray
            Array of shape (9, H) representing quantile levels [0.1..0.9].
        aux_quantiles : Optional[np.ndarray]
            Secondary quantile array (e.g., Timestral-209M) for uncertainty calibration.
        non_negative : bool
            Whether to clip negative values to 0.0.

        Returns
        -------
        np.ndarray
            Calibrated and monotonic quantile array of shape (9, H).
        """
        out = quantiles.copy()

        # If auxiliary quantiles are available, blend for sharp, calibrated uncertainty
        if aux_quantiles is not None and self.w_tms > 0:
            median = out[4]  # Preserve point prediction median
            out = (1.0 - self.w_tms) * out + self.w_tms * aux_quantiles
            # Shift back so median is anchored
            out[4] = median

        # Physical non-negativity
        if non_negative:
            out = np.maximum(out, 0.0)

        # Enforce strict quantile monotonicity
        out = np.sort(out, axis=0)

        return out
