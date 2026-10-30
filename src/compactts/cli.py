"""Command-line interface for CompactTS research and evaluation workflows."""

import argparse
import sys
import os
import json
import torch
import numpy as np

from compactts.utils.hardware import HardwareMonitor
from compactts.utils.profiler import EfficiencyProfiler
from compactts.models.compact_ts import CompactTS, CompactTSConfig
from compactts.data.synthetic import SyntheticTimeSeriesGenerator
from compactts.data.dataset import chronological_split, create_sliding_windows, TimeSeriesDataset
from compactts.training.trainer import CompactTSTrainer
from compactts.evaluation.benchmark import SmokeBenchmarkSuite
from compactts.evaluation.real_benchmark import RealGiftEvalBenchmarkSuite
from compactts.evaluation.leakage_checker import LeakageChecker


def cmd_system_check(args: argparse.Namespace) -> None:
    """Check and display system hardware and CUDA availability."""
    print("=" * 60)
    print(" COMPACTTS HARDWARE & ENVIRONMENT SYSTEM CHECK")
    print("=" * 60)
    info = HardwareMonitor.get_system_info()
    for k, v in info.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile model parameters, checkpoint size, latency, and memory."""
    tier = args.tier
    context_len = args.context_len
    pred_len = args.pred_len
    device = args.device

    print("=" * 60)
    print(f" PROFILING COMPACTTS (Tier: {tier}, Device: {device})")
    print("=" * 60)

    cfg = CompactTSConfig.from_tier(tier, context_len=context_len, pred_len=pred_len)
    model = CompactTS(cfg)

    param_info = EfficiencyProfiler.count_parameters(model)
    size_mb = EfficiencyProfiler.measure_checkpoint_size(model)
    bench_info = EfficiencyProfiler.benchmark_inference(
        model=model,
        context_len=context_len,
        device=device,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
    )

    print(f"  Trainable Parameters    : {param_info['trainable_parameters']:,}")
    print(f"  Total Parameters        : {param_info['total_parameters']:,}")
    print(f"  Checkpoint Disk Size    : {size_mb:.3f} MB")
    print(f"  Mean Latency (batch={args.batch_size}) : {bench_info['mean_latency_ms']:.3f} ms")
    print(f"  Median Latency (p50)    : {bench_info['median_latency_ms']:.3f} ms")
    print(f"  95th Percentile (p95)   : {bench_info['p95_latency_ms']:.3f} ms")
    print(f"  Throughput              : {bench_info['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Peak VRAM Allocation    : {bench_info['peak_vram_mb']:.2f} MB")
    print(f"  Process Resident RAM    : {bench_info['process_ram_mb']:.2f} MB")
    print("=" * 60)


def cmd_benchmark_smoke(args: argparse.Namespace) -> None:
    """Run local multi-domain smoke benchmark against baselines."""
    print("=" * 60)
    print(" RUNNING LOCAL MULTI-DOMAIN SMOKE BENCHMARK")
    print(f" Device: {args.device} | Epochs: {args.epochs}")
    print("=" * 60)

    suite = SmokeBenchmarkSuite(seed=args.seed)
    results = suite.run_benchmark(device=args.device, epochs=args.epochs)

    # Save machine-readable results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "smoke_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(" SMOKE BENCHMARK LEADERBOARD (GIFT-Eval MASE Rank)")
    print("=" * 60)
    for model_name, rank in results["rankings"]["MASE_Rank"].items():
        print(f"  {rank:5.2f} | {model_name}")
    print("=" * 60)
    print(f"Detailed machine-readable results saved to: {out_file}")


def cmd_benchmark_real(args: argparse.Namespace) -> None:
    """Run authentic benchmark on Salesforce/GiftEval datasets against baselines."""
    print("=" * 60)
    print(" RUNNING AUTHENTIC GIFT-EVAL BENCHMARK")
    print(f" Device: {args.device} | Epochs: {args.epochs}")
    print("=" * 60)

    runner = RealGiftEvalBenchmarkSuite()
    runner.run_benchmark(device=args.device, epochs=args.epochs, batch_size=args.batch_size)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a CompactTS configuration on synthetic or generated series."""
    print(f"Training CompactTS (Tier: {args.tier}, Epochs: {args.epochs})...")
    gen = SyntheticTimeSeriesGenerator(seed=args.seed)
    data = gen.generate_univariate(length=args.length)

    train_data, val_data, test_data = chronological_split(data, 0.7, 0.15)
    X_train, Y_train = create_sliding_windows(train_data, args.context_len, args.pred_len)
    X_val, Y_val = create_sliding_windows(val_data, args.context_len, args.pred_len)

    cfg = CompactTSConfig.from_tier(
        args.tier,
        context_len=args.context_len,
        pred_len=args.pred_len,
        head_type=args.head_type,
    )
    model = CompactTS(cfg)
    trainer = CompactTSTrainer(
        model,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, Y_val), batch_size=args.batch_size)

    summary = trainer.fit(train_loader, val_loader, epochs=args.epochs, model_name=f"compactts_{args.tier}.pt")
    print(f"Training completed in {summary['total_time_sec']}s. Best Val Loss: {summary['best_val_loss']:.5f}")
    print(f"Checkpoint saved to: {summary['checkpoint_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CompactTS CLI: State-of-the-Art Compact Forecasting Model")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # system-check
    subparsers.add_parser("system-check", help="Check hardware, RAM, and GPU/CUDA status")

    # profile
    p_profile = subparsers.add_parser("profile", help="Profile parameter count, memory, and inference latency")
    p_profile.add_argument("--tier", default="small", choices=["tiny", "small", "medium", "large"])
    p_profile.add_argument("--context-len", type=int, default=96)
    p_profile.add_argument("--pred-len", type=int, default=24)
    p_profile.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_profile.add_argument("--batch-size", type=int, default=1)
    p_profile.add_argument("--num-runs", type=int, default=50)

    # benchmark-smoke
    p_bench = subparsers.add_parser("benchmark-smoke", help="Run multi-domain smoke benchmark against baselines")
    p_bench.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_bench.add_argument("--epochs", type=int, default=10)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--output-dir", default="reports")

    # benchmark-real
    p_real = subparsers.add_parser("benchmark-real", help="Run real GIFT-Eval benchmark against baselines")
    p_real.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_real.add_argument("--epochs", type=int, default=12)
    p_real.add_argument("--batch-size", type=int, default=64)

    # train
    p_train = subparsers.add_parser("train", help="Train a CompactTS model")
    p_train.add_argument("--tier", default="small", choices=["tiny", "small", "medium", "large"])
    p_train.add_argument("--context-len", type=int, default=96)
    p_train.add_argument("--pred-len", type=int, default=24)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--head-type", default="point", choices=["point", "quantile", "distribution"])
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--length", type=int, default=2000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")

    args = parser.parse_args()

    if args.command == "system-check":
        cmd_system_check(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "benchmark-smoke":
        cmd_benchmark_smoke(args)
    elif args.command == "benchmark-real":
        cmd_benchmark_real(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
