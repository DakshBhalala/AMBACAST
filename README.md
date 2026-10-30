# CompactTS-Agent: Autonomous Agentic Time-Series Meta-Forecaster

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-purple.svg)](LICENSE)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-FFD21E)](https://huggingface.co/BhalalaDaksh/CompactTS-Agent)

**CompactTS-Agent** is an autonomous agentic meta-forecasting system developed by **Daksh Bhalala**. Designed for general time-series forecasting across diverse industrial domains (energy grids, server telemetry, traffic flow, nature observations, and healthcare), it dynamically extracts structural spectral/seasonal characteristics and routes forecasting tasks across specialized foundation model backbones.

---

## Architecture Overview

```
       Incoming Time Series History y[1:T]
                         │
                         ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 1. Autonomous Scale-Invariant Profiler (Feature Extractor)│
   │    • Fourier Spectral Entropy (Hs: harmonic purity)       │
   │    • Normalized Distribution Shift (Delta_mu: drift)      │
   │    • ARCH Volatility Clustering (R^2_ARCH: heterosked.)   │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                     Scale-Invariant Signatures
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 2. The Meta-Router (The "Brain")                          │
   │    Learned physical decision boundary via LODO validation  │
   │    mapping series regimes to the optimal foundation expert│
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                         Routing Decision
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ 3. Specialized Foundation Model Backbones                 │
   │    • Concentrated harmonics -> Harmonic Fourier basis     │
   │    • Low-drift telemetry -> Chronos-Bolt                  │
   │    • Volatility clustering -> Deep Autoregressive Context │
   │    • Non-negative support projection & quantile sort      │
   └───────────────────────────────────────────────────────────┘
```

---

## ⚡ Installation & Setup

```bash
git clone https://github.com/DakshBhalala/CompactTS-Agent.git
cd CompactTS-Agent
pip install -e .
```

---

## 📊 Benchmark Evaluation & Policy Simulation

### Option A: Offline Meta-Routing Policy Simulation
Demonstrates the autonomous meta-router's dataset profiling and foundation expert selection across benchmark configurations:

```bash
# Run meta-routing policy simulation across configurations
python scripts/run_agentic_benchmark.py

# Validate submission folder schema
python scripts/validate_results.py results/CompactTS-Agent
```

*Note: In accordance with GIFT-Eval leaderboard guidelines for proprietary/agentic harnesses (e.g. Google's STRIDE and RegLab-AH-2), `replication_code_available` is registered as `"No"` for this submission because the end-to-end multi-model orchestration harness is governed under BSL 1.1.*

### Option B: Python API Usage

```python
import numpy as np
import pandas as pd
from compactts.agent import CompactTSAgent, SeriesProfiler

# 1. Profile an incoming time series
profiler = SeriesProfiler(freq="H")
series = np.sin(np.linspace(0, 20, 100)) + 0.1 * np.random.randn(100)
profile = profiler.profile(series)
print("Series Profile:", profile)

# 2. Generate calibrated probabilistic forecasts
agent = CompactTSAgent(prediction_length=24, freq="H")
start_dates = [pd.Period("2024-01-01 00:00", freq="h")]
forecasts = agent.forecast_batch([series], start_dates)

# Access forecast quantiles (9 levels: 0.1 to 0.9)
q_forecasts = forecasts[0].forecast_array
print("Forecast shape (9 quantiles, 24 horizon):", q_forecasts.shape)
```

---

## 🧪 Unit Tests

Run the test suite:
```bash
pytest
```

---

## 💼 Commercial Licensing & Enterprise Deployment

CompactTS-Agent is licensed under the **Business Source License 1.1 (BSL 1.1)**:
- **Free for Research, Academic Study, Evaluation, & Benchmark Replication:** Anyone is free to inspect, run, evaluate, test, and reproduce benchmarks using CompactTS-Agent.
- **Commercial & Production Use:** Any commercial use, deployment in production environments, hosting as a paid service/API, or integration into commercial products requires a separate commercial license agreement.

For commercial inquiries, enterprise deployment rights, or proprietary licensing, contact:
- **Author:** Daksh Bhalala
- **GitHub:** [DakshBhalala/CompactTS-Agent](https://github.com/DakshBhalala/CompactTS-Agent)

---

## 📋 Project Information

- **Author / Organization:** Daksh Bhalala
- **Hugging Face Model:** [BhalalaDaksh/CompactTS-Agent](https://huggingface.co/BhalalaDaksh/CompactTS-Agent)
- **Official GIFT-Eval PR:** [SalesforceAIResearch/gift-eval#242](https://github.com/SalesforceAIResearch/gift-eval/pull/242)
- **License:** Business Source License 1.1 (BSL-1.1)
