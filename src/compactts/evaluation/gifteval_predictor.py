"""Official GluonTS-compatible Predictor for CompactTS.

Conforms to the Salesforce GIFT-Eval benchmark predictor interface, wrapping
CompactTS inside GluonTS's Predictor abstraction to generate calibrated SampleForecasts.
"""

from typing import Iterable, Iterator, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.dataset import DataEntry

from compactts.models.compact_ts import CompactTS, CompactTSConfig


def impute_series(x: np.ndarray) -> np.ndarray:
    """Forward and backward fill NaNs; all-NaN series become zeros."""
    if not np.isnan(x).any() and not np.isinf(x).any():
        return x.astype(np.float32)
    s = pd.Series(x.astype(np.float64)).ffill().bfill()
    return np.nan_to_num(s.to_numpy(), nan=0.0).astype(np.float32)


def compute_series_stats(y: np.ndarray) -> tuple[float, float]:
    """Compute robust mean and standard deviation for normalization fallback."""
    valid = y[np.isfinite(y)]
    if valid.size == 0:
        return 0.0, 1.0
    mu = float(valid.mean())
    sd = float(valid.std())
    return mu, (sd if sd > 1e-8 else 1.0)


def left_pad_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    """Pad sequence with leftmost value or zeros if shorter than target_len."""
    curr_len = len(x)
    if curr_len >= target_len:
        return x[-target_len:]
    pad_len = target_len - curr_len
    pad_val = x[0] if curr_len > 0 else 0.0
    return np.pad(x, (pad_len, 0), mode="constant", constant_values=pad_val)


class CompactTSPredictor(Predictor):
    """GluonTS Predictor wrapper for CompactTS and CompactTSPro models."""

    def __init__(
        self,
        model: Any,
        context_len: Optional[int] = None,
        prediction_length: Optional[int] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        num_samples: int = 100,
        batch_size: int = 128,
        clip_val: float = 20.0,
        **kwargs: Any,
    ) -> None:
        c_len = context_len or kwargs.get("context_length") or getattr(getattr(model, "config", None), "context_len", 96)
        p_len = prediction_length or getattr(getattr(model, "config", None), "pred_len", 24)
        super().__init__(prediction_length=p_len)
        self.device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.model = model.eval().to(self.device)
        self.context_len = c_len
        self.H = p_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.n_nonfinite_replaced = 0

    def _sample_model(self, x_tensor: torch.Tensor, target_h: int) -> np.ndarray:
        """Sample from model with support for arbitrary horizon target_h."""
        model_pred_len = getattr(getattr(self.model, "config", None), "pred_len", 24)

        if hasattr(self.model, "sample"):
            if target_h <= model_pred_len:
                with torch.no_grad():
                    smp_t = self.model.sample(x_tensor, self.num_samples)
                    if smp_t.shape[-1] == 1:
                        smp_t = smp_t.squeeze(-1)
                    return smp_t[:, :, :target_h].cpu().numpy()

            # Autoregressive multi-horizon roll-out for target_h > model_pred_len
            chunks = []
            curr_x = x_tensor.clone()
            steps = (target_h + model_pred_len - 1) // model_pred_len
            with torch.no_grad():
                for step_idx in range(steps):
                    smp_t = self.model.sample(curr_x, self.num_samples)
                    if smp_t.shape[-1] == 1:
                        smp_t = smp_t.squeeze(-1)
                    chunks.append(smp_t)
                    mean_step = smp_t.mean(dim=1, keepdim=True).permute(0, 2, 1)
                    if step_idx > 0:
                        decay = 0.98 ** step_idx
                        mean_step = mean_step * decay
                    if curr_x.shape[1] >= model_pred_len:
                        curr_x = torch.cat([curr_x[:, model_pred_len:, :], mean_step], dim=1)
                    else:
                        curr_x = torch.cat([curr_x, mean_step], dim=1)[:, -self.context_len :, :]
            full_smp = torch.cat(chunks, dim=-1)
            return full_smp[:, :, :target_h].cpu().numpy()

        elif self.model.config.head_type == "distribution":
            mean_t, std_t = self.model(x_tensor)
            eps = torch.randn(self.num_samples, *mean_t.shape, device=self.device)
            smp_t = (mean_t.unsqueeze(0) + std_t.unsqueeze(0) * eps).squeeze(-1).permute(1, 0, 2)
            raw = smp_t.cpu().numpy()
            if raw.shape[-1] > target_h:
                raw = raw[..., :target_h]
            return raw
        elif self.model.config.head_type == "quantile":
            q_out = self.model(x_tensor).squeeze(2)
            med = q_out[..., q_out.shape[-1] // 2]
            q_min = q_out[..., 0]
            q_max = q_out[..., -1]
            scale = (q_max - q_min) / 3.29
            eps = torch.randn(self.num_samples, *med.shape, device=self.device)
            smp_t = (med.unsqueeze(0) + scale.unsqueeze(0) * eps).permute(1, 0, 2)
            raw = smp_t.cpu().numpy()
            if raw.shape[-1] > target_h:
                raw = raw[..., :target_h]
            return raw
        else:
            point_pred = self.model(x_tensor).squeeze(-1)
            sigma = torch.clamp(point_pred.abs() * 0.05, min=0.01)
            eps = torch.randn(self.num_samples, *point_pred.shape, device=self.device)
            smp_t = (point_pred.unsqueeze(0) + sigma.unsqueeze(0) * eps).permute(1, 0, 2)
            raw = smp_t.cpu().numpy()
            if raw.shape[-1] > target_h:
                raw = raw[..., :target_h]
            return raw

    def predict(self, dataset: Iterable[DataEntry], **kwargs: Any) -> Iterator[SampleForecast]:
        """Generate forecasts for each time-series entry in dataset."""
        entries = list(dataset)
        B = self.batch_size

        for k in range(0, len(entries), B):
            chunk = entries[k : k + B]
            X_batch = []
            stats_list = []
            meta_list = []

            for e in chunk:
                y = np.asarray(e["target"], dtype=np.float32)
                # Handle univariate or multivariate entry
                if y.ndim == 2:
                    y = y[0]  # Select primary variate if univariate forecast requested

                mu, sd = compute_series_stats(y)
                cleaned = impute_series(y)
                # RevIN handles normalization inside CompactTS, but standardizing avoids extreme sensor spikes
                standardized = np.clip((cleaned - mu) / sd, -self.clip_val, self.clip_val)
                padded = left_pad_sequence(standardized, self.context_len)

                X_batch.append(padded)
                stats_list.append((mu, sd))
                # Next timestamp after historical target
                meta_list.append((e["start"] + len(y), e.get("item_id")))

            # Shape: (batch_size, context_len, 1)
            x_tensor = torch.from_numpy(np.stack(X_batch)).unsqueeze(-1).to(self.device)

            raw_samples = self._sample_model(x_tensor, self.H)

            # Denormalize and yield SampleForecast
            for b_idx, ((mu, sd), (start_date, item_id)) in enumerate(zip(stats_list, meta_list)):
                samples = raw_samples[b_idx] * sd + mu
                bad = ~np.isfinite(samples)
                if bad.any():
                    self.n_nonfinite_replaced += int(bad.sum())
                    samples = np.where(bad, mu, samples)

                yield SampleForecast(samples=samples, start_date=start_date, item_id=item_id)
