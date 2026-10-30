"""Script to profile all CompactTS model parameter tiers."""

import torch
from compactts.models.compact_ts import CompactTS, CompactTSConfig
from compactts.utils.profiler import EfficiencyProfiler

def main() -> None:
    tiers = ['tiny', 'small', 'medium', 'large']
    print(f"{'Tier':<8} | {'Params':<10} | {'Disk Size':<10} | {'GPU Latency':<12} | {'CPU Latency':<12} | {'GPU Peak VRAM':<12}")
    print("-" * 75)
    for tier in tiers:
        cfg = CompactTSConfig.from_tier(tier)
        model = CompactTS(cfg)
        params = EfficiencyProfiler.count_parameters(model)
        size_mb = EfficiencyProfiler.measure_checkpoint_size(model)
        bench_gpu = EfficiencyProfiler.benchmark_inference(
            model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=32,
            num_runs=30
        )
        bench_cpu = EfficiencyProfiler.benchmark_inference(
            model,
            device='cpu',
            batch_size=1,
            num_runs=30
        )
        p_count = f"{params['trainable_parameters']:,}"
        s_mb = f"{size_mb:.3f} MB"
        gpu_lat = f"{bench_gpu['mean_latency_ms']:.2f} ms"
        cpu_lat = f"{bench_cpu['mean_latency_ms']:.2f} ms"
        gpu_vram = f"{bench_gpu['peak_vram_mb']:.2f} MB"
        print(f"{tier.upper():<8} | {p_count:<10} | {s_mb:<10} | {gpu_lat:<12} | {cpu_lat:<12} | {gpu_vram:<12}")

if __name__ == "__main__":
    main()
