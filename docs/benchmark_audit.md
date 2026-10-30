# GIFT-Eval Benchmark Protocol & Data Integrity Audit

**Audit Date:** October 20, 2026  
**Auditor:** Daksh Bhalala  
**Target Repository:** CompactTS (`d:\EXPOSE\AMBACAST`)  
**Status:** COMPLETE — Critical Deviations Identified & Corrected

---

## 1. Executive Summary of Audit Findings

| Component | Previous Smoke-Test Implementation | Official GIFT-Eval Protocol | Audit Status | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| **Data Source** | `SyntheticTimeSeriesGenerator` (Synthetic series) | 28 official Arrow tables hosted at `Salesforce/GiftEval` on Hugging Face | **DEVIATION** | Replace synthetic suites with genuine `Salesforce/GiftEval` Arrow tables |
| **Dataset Naming** | `energy_hourly`, `weather_daily`, `traffic_15min`, `econ_daily` (Synthetic) | `ett1/H`, `jena_weather/H`, `us_births/D`, `SZ_TAXI/H`, etc. | **DEVIATION** | Clearly label earlier results as *Stage A Synthetic Smoke Test* |
| **Splitting Strategy** | Chronological 70% Train, 15% Val, 15% Test | Benchmark rolling evaluation windows / fixed forecast horizons | **ACCEPTABLE (In-Domain)** | Maintain strict chronological splits with explicit zero-leakage verification |
| **Evaluation Mode** | In-domain supervised training (Objective A) | Zero-shot evaluation across unseen datasets (Objective B) | **DISTINCTION REQUIRED** | Explicitly distinguish per-dataset predictor results from general zero-shot foundation capabilities |
| **Metric Calculation** | Exact GIFT-Eval formulas: MASE (seasonal naive denom), CRPS, WQL, sMAPE | Official GIFT-Eval definitions for MASE, CRPS, WQL | **VERIFIED CORRECT** | Preserved without change |
| **Normalization** | Global `SafeStandardScaler` fit on train split | Window-level Reversible Instance Normalization (RevIN) | **REFINED** | Let CompactTS RevIN operate directly without external data pre-scaling distortions |
| **Baseline Fairness** | Identical context/horizon, identical splits, metrics computed on raw scale | Uniform inputs and targets across all competitive baselines | **VERIFIED CORRECT** | All statistical and neural baselines evaluated under exact identical conditions |

---

## 2. Detailed Audit of the Previous Smoke Test

### 2.1 Synthetic vs. Real Benchmark Datasets
In the initial development cycle (Stage A), `SmokeBenchmarkSuite` in [`src/compactts/evaluation/benchmark.py`](file:///d:/EXPOSE/AMBACAST/src/compactts/evaluation/benchmark.py) instantiated synthetic time series using mathematical generators (e.g. sinusoidal harmonics, linear trends, autoregressive shocks).
- **The Finding:** While these series allowed rapid end-to-end debugging of shapes, loss functions, metrics, and trainer loops, **they were synthetic approximations and NOT the official test sets of GIFT-Eval**.
- **Corrective Policy:** Under no circumstances will synthetic benchmark scores be claimed as official GIFT-Eval leaderboard results. All official benchmark numbers must be derived directly from authentic data downloaded from `Salesforce/GiftEval`.

### 2.2 Objective A (Compact Predictor) vs. Objective B (Foundation Model)
GIFT-Eval was primarily designed to evaluate **zero-shot generalization** of multi-billion parameter foundation models (e.g., Chronos-Large 710M, TimesFM 200M/500M, Moirai-1.0-R).
- In our current Stage A and Stage B research, CompactTS is evaluated under **Objective A: Compact Domain Predictor** (training on 70% of a domain's history and forecasting the held-out 15% future).
- This is a legitimate forecasting research objective, but must not be conflated with the zero-shot leaderboard where models have never seen the target series during training.
- We establish a clear dual-track evaluation:
  1. **Track 1 (In-Domain Benchmark):** CompactTS vs DLinear, NLinear, Ridge, LightGBM, Seasonal Naive, AutoETS, and AutoARIMA on held-out test splits.
  2. **Track 2 (Zero-Shot Transferability):** Cross-dataset zero-shot evaluation (training on source datasets, evaluating on unseen target datasets without weight updates).

### 2.3 Split Integrity and Normalization Causality
- The chronological splitting in `chronological_split` enforces $T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$ with zero temporal overlap.
- `LeakageChecker.verify_scaler_integrity` verifies that scaler parameters ($\mu, \sigma$) are computed exclusively on $X_{\text{train}}$.
- When passing data to CompactTS, the architecture's native `RevIN` component handles instance-level non-stationarity dynamically.

### 2.4 Metric Mathematical Compliance
The metric engine in [`src/compactts/evaluation/metrics.py`](file:///d:/EXPOSE/AMBACAST/src/compactts/evaluation/metrics.py) was checked against official GIFT-Eval definitions:
- **MASE (Mean Absolute Scaled Error):**
  $$\text{MASE} = \frac{\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|}{\frac{1}{T-m} \sum_{t=m+1}^T |y_t - y_{t-m}|}$$
  where the denominator is calculated over the in-sample training series with seasonality $m$.
- **WQL (Weighted Quantile Loss):**
  $$\text{WQL}_q = 2 \frac{\sum_t \max(q(y_t - \hat{y}_{t,q}), (1-q)(\hat{y}_{t,q} - y_t))}{\sum_t |y_t|}$$
- **CRPS (Continuous Ranked Probability Score):**
  Calculated across the 9 standard evaluation quantiles ($q \in \{0.1, 0.2, \dots, 0.9\}$) via numerical quadrature approximation.
- **Leaderboard Ranking:**
  Models are ranked per dataset by MASE and CRPS (1 = lowest error), and ranks are averaged across all benchmark datasets.

---

## 3. Authentic GIFT-Eval Dataset Subset Specification

For Stage B real-world evaluation, 4 official datasets are ingested from `Salesforce/GiftEval` via `GiftEvalDataLoader`:

| Dataset Identifier | Domain | Frequency | Sampling Seasonality ($m$) | Context Horizon ($L$) | Forecast Horizon ($H$) | Authentic Series Size | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ett1_hourly` | Energy | Hourly (1H) | 24 | 96 | 24 | 17,420 steps (7 channels, target: `OT`) | Apache 2.0 |
| `jena_weather_hourly` | Nature/Weather | Hourly (1H) | 24 | 96 | 24 | 8,784 steps (21 channels, target: `T (degC)`) | Apache 2.0 |
| `us_births_daily` | Healthcare/Demographics | Daily (1D) | 7 | 60 | 14 | 7,305 steps (1 channel) | Apache 2.0 / Public Domain |
| `sz_taxi_hourly` | Transport | Hourly (1H) | 24 | 96 | 24 | 744 steps (156 sensors) | Apache 2.0 |

All raw data files are PyArrow IPC streams directly mirrored from the official benchmark repository.
