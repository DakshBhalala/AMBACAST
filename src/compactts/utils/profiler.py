"""Benchmarking and profiling suite for forecasting models.

Measures parameter count, checkpoint size, latency, throughput, and memory footprint.
"""

from typing import Dict, Any, Tuple, Optional
import os
import tempfile
import time
import torch
import torch.nn as nn
import numpy as np

from compactts.utils.hardware import HardwareMonitor


class EfficiencyProfiler:
    """Profiles computational efficiency of forecasting models on CPU and GPU."""

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and non-trainable parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "non_trainable_parameters": total - trainable,
        }

    @staticmethod
    def measure_checkpoint_size(model: nn.Module) -> float:
        """Measure serialized PyTorch checkpoint size in megabytes (MB)."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name
        try:
            torch.save(model.state_dict(), temp_path)
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return round(size_mb, 3)

    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        context_len: int = 96,
        num_features: int = 1,
        device: str = "cpu",
        batch_size: int = 1,
        num_warmup: int = 10,
        num_runs: int = 50,
    ) -> Dict[str, Any]:
        """Benchmark forward pass latency and throughput.

        Args:
            model: PyTorch forecasting model.
            context_len: Input history length.
            num_features: Number of time-series variates.
            device: 'cpu' or 'cuda'.
            batch_size: Batch size for testing.
            num_warmup: Warmup iterations.
            num_runs: Measurement iterations.

        Returns:
            Dictionary with latency (ms), throughput (samples/sec), and memory.
        """
        device_obj = torch.device(device)
        model = model.to(device_obj).eval()

        dummy_input = torch.randn(batch_size, context_len, num_features, device=device_obj)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()

        HardwareMonitor.reset_peak_memory()

        # Timed benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

        latencies_arr = np.array(latencies)
        mean_latency = float(np.mean(latencies_arr))
        p50_latency = float(np.median(latencies_arr))
        p95_latency = float(np.percentile(latencies_arr, 95))
        throughput = float((batch_size / (mean_latency / 1000.0)))

        peak_mem = HardwareMonitor.get_peak_memory()

        return {
            "device": device,
            "batch_size": batch_size,
            "mean_latency_ms": round(mean_latency, 3),
            "median_latency_ms": round(p50_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "peak_vram_mb": peak_mem["peak_vram_mb"],
            "process_ram_mb": peak_mem["process_ram_mb"],
        }
