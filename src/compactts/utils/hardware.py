"""Hardware inspection and memory monitoring utilities.

Tracks GPU VRAM, system RAM, CPU cores, and hardware constraints.
"""

from typing import Dict, Any, Optional
import os
import sys
import psutil
import torch


class HardwareMonitor:
    """Monitors system resources (RAM, VRAM, CPU, GPU)."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect comprehensive system hardware specifications."""
        ram_info = psutil.virtual_memory()
        info: Dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(ram_info.total / (1024 ** 3), 2),
            "ram_available_gb": round(ram_info.available / (1024 ** 3), 2),
            "ram_percent_used": ram_info.percent,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_total_gb"] = round(vram_bytes / (1024 ** 3), 2)
            info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
        else:
            info["gpu_name"] = "None (CPU only)"
            info["vram_total_gb"] = 0.0

        return info

    @staticmethod
    def reset_peak_memory() -> None:
        """Reset peak memory tracking on CUDA if available."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def get_peak_memory() -> Dict[str, float]:
        """Return peak memory usage since last reset."""
        result = {}
        if torch.cuda.is_available():
            result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
            result["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2)
        else:
            result["peak_vram_mb"] = 0.0
            result["peak_vram_reserved_mb"] = 0.0

        process = psutil.Process(os.getpid())
        result["process_ram_mb"] = round(process.memory_info().rss / (1024 ** 2), 2)
        return result
