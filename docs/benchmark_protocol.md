# GIFT-Eval Benchmark Protocol & Smoke Evaluation Suite

**Document:** Evaluation Protocol & Benchmarking Guidelines  
**Project:** CompactTS  
**Date:** October 20, 2026  
**Primary Standards:** [SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval) & [tsfm.ai/benchmarks/gift-eval](https://tsfm.ai/benchmarks/gift-eval)

---

## 1. Official GIFT-Eval Protocol Compliance

GIFT-Eval is the official benchmark for evaluating general-purpose and zero-shot time-series forecasting foundation models. To ensure legitimate scientific comparison and reproducibility, CompactTS adheres strictly to the benchmark's core guidelines:

1. **No Data Leakage / Contamination:**
   - Evaluated models must not have seen the test horizons of evaluation datasets during pretraining or parameter tuning.
   - All input scalers, normalizers, and feature estimators must be computed solely using the in-sample training partition.
2. **Strict Chronological Splits:**
   - Shuffling time-series observations is strictly prohibited.
   - Splits must preserve temporal continuity: $t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$.
3. **Multi-Domain & Multi-Frequency Coverage:**
   - The official suite evaluates 97 configurations across 7 domains and 10 sampling frequencies.
   - The local smoke suite mirrors these multi-scale dynamics across 4 representative domains.
4. **Ranking by Average Rank:**
   - In accordance with GIFT-Eval, models are ranked per dataset configuration. The overall leaderboard metric is the average rank across all tasks (`MASE_Rank` and `CRPS_Rank`), preventing scale distortion from volatile outlier series.

---

## 2. Multi-Domain Smoke Benchmark Suite

Due to the computational and storage intensity of downloading the full 177-million-point GIFT-Eval corpus onto local consumer hardware, we establish the **Stage A: Local Smoke Benchmark Suite**.

This suite tests the 4 primary structural challenges represented in GIFT-Eval:

| Scenario Name | Domain | Sampling Frequency | Seasonality ($m$) | Context $L$ | Horizon $H$ | Primary Structural Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`energy_hourly`** | Energy | 1H (Hourly) | 24 (Daily) | 96 | 24 | Strong dual seasonality (daily 24, weekly 168), linear load growth, Gaussian noise. |
| **`weather_daily`** | Nature / Weather | 1D (Daily) | 7 (Weekly) | 60 | 14 | Annual macro cycle (365), weekly synoptic cycles (7), structural regime change points. |
| **`traffic_15min`** | Transport | 15T (15-Minute) | 96 (Daily) | 192 | 48 | High-frequency intra-day peak traffic rushes (96 steps/day), weekend shifts (672 steps/week). |
| **`econ_daily`** | Economics / Finance | 1D (Daily) | 5 (Trading Week) | 60 | 10 | Stochastic trend drift, autoregressive momentum (AR(2)), heavy-tailed Student-t volatility ($df=4$). |

---

## 3. Strict Anti-Leakage Verification Suite

To independently audit pipeline integrity, every dataset evaluated through CompactTS must pass 4 programmatic leakage checks:

1. **Split Monotonicity Check:**
   $$\max(\text{Index}_{\text{train}}) < \min(\text{Index}_{\text{val}}) < \max(\text{Index}_{\text{val}}) < \min(\text{Index}_{\text{test}})$$
2. **Scaler Independence Check:**
   Scaler parameters $(\mu_{\text{scaler}}, \sigma_{\text{scaler}})$ must match $\text{train}$ statistics within numerical tolerance ($< 10^{-6}$) and differ from the global series statistics.
3. **Window Boundary Isolation:**
   The training target window $Y_{\text{train}}$ must terminate strictly before or at the start of the evaluation context window $X_{\text{test}}$.
4. **Context-Target Separation:**
   Context windows $X$ must not contain future ground-truth values from $Y$.

---

## 4. Submission Schema & Output Format

Submissions to the official GIFT-Eval leaderboard generate an entry folder `results/<MODEL_NAME>/` containing:

### 4.1 `config.json`
```json
{
  "model_name": "CompactTS-Small",
  "model_type": "deep-learning",
  "parameters": 745820,
  "test_data_leakage": false,
  "zero_shot": false,
  "code_link": "https://github.com/DakshBhalala/AMBACAST"
}
```

### 4.2 `all_results.csv`
Contains evaluation rows adhering to the official schema:
```csv
dataset,model,eval_metrics/MASE[0.5],eval_metrics/CRPS,eval_metrics/MSE[mean],eval_metrics/MAE[mean],domain,num_variates
energy_hourly/1H/short,CompactTS-Small,0.482,0.198,0.312,0.421,Energy,1
weather_daily/1D/short,CompactTS-Small,0.615,0.284,0.540,0.582,Nature,1
traffic_15min/15T/short,CompactTS-Small,0.512,0.210,0.395,0.470,Transport,1
econ_daily/1D/short,CompactTS-Small,0.892,0.415,0.781,0.720,Economics,1
```
