# Comprehensive Time-Series Forecasting Research Review & Ecosystem Survey

**Project:** CompactTS (State-of-the-Art Compact Time-Series Forecasting Model)  
**Date:** October 20, 2026  
**Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), 16 GB System RAM, Windows 11, Python 3.12, PyTorch 2.5.1+cu121.  
**Primary Benchmark:** GIFT-Eval ([SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval)) & [TSFM.ai Leaderboard](https://tsfm.ai/benchmarks/gift-eval).

---

## 1. Executive Summary & Research Gap

Modern time-series forecasting has undergone a paradigm shift from dataset-specific statistical models (ARIMA, ETS) toward Time Series Foundation Models (TSFMs). However, the current landscape exhibits a severe polarization:

1. **Massive Transformer & LLM-Derived Models (100M – 710M+ parameters):**
   - Models such as Google TimesFM (200M–500M), Amazon Chronos-T5 (up to 710M), Chronos-2 (120M), and Mistral Timestral-209M achieve high zero-shot generalization across diverse domains.
   - **Critical Drawbacks:** Heavy parameter footprint, large GPU VRAM requirements during inference (2–6+ GB), slow autoregressive decoding or quadratic self-attention latency, and high deployment costs. Furthermore, several weights are published under restrictive commercial licenses (e.g. TimesFM 3.0 non-commercial, Timestral Modified MIT with a $20M monthly revenue cap).
2. **Minimalist Linear & Decomposition Models (1k – 50k parameters):**
   - DLinear and NLinear demonstrated that simple single-layer linear mappings with moving-average trend decomposition can outperform complex Transformers on traditional academic benchmarks (ETT, Weather, Electricity).
   - **Critical Drawbacks:** Pure linear mappings lack representation capacity for nonlinear multi-frequency interactions, cannot effectively share deep representations across diverse cross-domain datasets, fail under sudden regime shifts or heavy-tailed stochastic volatility, and lack native probabilistic calibration.
3. **The Compact Frontier & Research Gap:**
   - Recent breakthroughs such as IBM's Tiny Time Mixers (TTM, ~1M params), TW3Cast (frozen router without LLMs), and NX-AI TiRex (35M params) prove that **scale is not all you need** in time-series forecasting. With the right inductive biases—namely, reversible instance normalization (RevIN), multi-scale temporal decomposition, multi-kernel depthwise dilated convolutions, and gated residual mixing—a model with **under 1 million parameters** can achieve competitive accuracy while running orders of magnitude faster on consumer laptop GPUs (6GB VRAM) and CPUs.

---

## 2. Primary Benchmark: GIFT-Eval Deep Dive

### 2.1 Benchmark Architecture & Scope
GIFT-Eval (*General TIme Series ForecasTing Model Evaluation*), introduced by Salesforce AI Research (Aksu et al., 2024), is the gold standard for zero-shot and full-shot evaluation:
- **Datasets:** 23 to 28 core datasets encompassing 97 distinct evaluation configurations.
- **Scale:** Over 144,000 individual time series and 177 million data points.
- **Domains (7 categories):** Energy, Transport, Weather/Nature, Healthcare/Demographics, Web/CloudOps, Sales/Retail, and Economics/Finance.
- **Temporal Frequencies (10 types):** From sub-minute (10S, 5T, 10T, 15T) to hourly (1H), daily (1D), weekly (1W), monthly (1M), quarterly (1Q), and yearly (1A).
- **Variates & Horizons:** Both univariate and multivariate time series with short, medium, and long forecast horizons.
- **Leakage Prevention:** Accompanied by `GiftEvalPretrain` (~230 billion observations) which strictly excludes test portions of downstream benchmark datasets to guarantee contamination-free zero-shot evaluation.

### 2.2 Official Evaluation Protocol & Metrics
Evaluation conforms to GluonTS standards with `mask_invalid_label=True` and `axis=None`:
1. **Continuous Ranked Probability Score (CRPS):**
   $$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}_{z \ge y})^2 dz$$
   Approximated discretely via multi-quantile pinball loss across quantiles $q \in \{0.1, 0.2, \dots, 0.9\}$.
2. **Mean Absolute Scaled Error (MASE):**
   $$\text{MASE} = \frac{\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|}{\frac{1}{T - m} \sum_{t=m+1}^T |y_t - y_{t-m}|}$$
   Evaluates point forecast accuracy relative to an in-sample seasonal naive baseline of period $m$. A value $< 1.0$ indicates that the model outperforms the seasonal naive baseline.
3. **Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Weighted Quantile Loss (WQL).**

### 2.3 Official Aggregation & Ranking Scheme
To prevent individual high-variance datasets from skewing raw arithmetic averages:
- Models are ranked independently on each dataset slice.
- **`MASE_Rank`** is the arithmetic mean of the model's rank across all dataset configurations for point forecasts.
- **`CRPS_Rank`** is the arithmetic mean of the model's rank across all dataset configurations for probabilistic forecasts.
- The leaderboard classifies submissions under `model_type` (`statistical`, `deep-learning`, `pretrained`, `agentic`), explicitly tracking `test_data_leakage` (boolean flag) and requiring replication code.

---

## 3. Comprehensive Model Survey

| Model Family | Organization | Architecture Family | Parameter Count | Checkpoint Size | Context / Horizon | Inference Compute / VRAM | Code & Weight License | Strengths | Known Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STRIDE w/ Synapse** | Google Cloud AI Research | Agentic + Cross-Modal Reasoning Prior | ~100M – 300M | ~400 – 1200 MB | 512–2048 / 96–720 | High (4–6 GB VRAM) | Research / Apache 2.0 (code) | Top MASE rank on GIFT-Eval; qualitative reasoning injection | Heavy runtime latency; high compute cost |
| **Timestral-209M** | Mistral AI | Patch-based Transformer (Time + Variate Attn) | 209M (also 5M, 28M, 123M) | ~800 MB (fp16) | Up to 32k / 1024 | Medium-High (2–4 GB VRAM) | Code: Apache 2.0; Weights: Modified MIT ($20M cap) | SOTA zero-shot probabilistic forecasting; 21 quantiles; covariates | Prohibitive commercial clause for large firms; high memory for 209M |
| **TimesFM (1.0/2.5/3.0)** | Google Research | Decoder-Only Transformer (Patch 32 -> 128) | 200M (v1/v2.5), 330M (v3), 500M | ~800 MB (v2.5) | 512 / 128–512 | Medium (2–3 GB VRAM) | v1/v2.5: Apache 2.0; v3: Non-Commercial | Fast zero-shot patch decoding; strong univariate foundation | v3 weights non-commercial; fixed patch granularity |
| **Chronos & Chronos-Bolt** | Amazon Science | T5 Encoder-Decoder / Direct Head | 8M (Tiny) to 710M (Large) | 35 MB (Tiny) – 2.8 GB (Large) | 512 / 64–256 | Tiny: <100 MB; Base: ~1 GB; Large: ~4 GB | Apache 2.0 (code + weights) | Permissive open source; Bolt is 250x faster with patch head | Original T5 tokenization slow; fixed discrete 4096 bins |
| **Chronos-2** | Amazon Science | Unified Encoder-Only Patch Transformer | 120M | ~480 MB (fp16) | 1024 / 256 | Medium (~1.5 GB VRAM) | Apache 2.0 | Native multivariate, covariates, and probabilistic output | Higher inference compute than compact linear/MLP models |
| **Moirai / Moirai-MoE** | Salesforce AI Research | Multi-Patch Encoder Transformer / Sparse MoE | 14M (Small), 91M (Base), 311M (Large) | ~56 MB – 1.2 GB | Multi-patch (8, 16, 32, 64, 128) | 500 MB – 3 GB VRAM | Custom Non-Commercial Research License | Any-variate attention handles arbitrary channel counts | Non-commercial license prevents commercial exploitation |
| **TiRex / TiRex-2** | NX-AI | xLSTM (Extended LSTM) Recurrent Backbone | 35M (v1), 38.4M active / 82.5M total (v2) | ~150 – 330 MB | Linear context scaling | Low-Medium (stable VRAM footprint) | TiRex-1: NXAI Community; TiRex-2: Apache 2.0 | $O(N)$ linear memory scaling; edge and PLC friendly | Custom xLSTM kernels require specialized GPU environments |
| **TW3Cast** | TW3 Partners | Frozen Training-Split Router + Fine-Tuned Models | Composite (~30M – 120M) | Variable | Dataset dependent | Medium | Code: Open; Papers: arXiv perpetual | 3rd place on GIFT-Eval without runtime LLM agents | Complex multi-checkpoint ensemble routing pipeline |
| **OpenTSLM-TeeMoE** | Stanford / ETH Zurich | LoRA MoE over Language Model Backbone | ~200M – 1B | ~800 MB – 4 GB | 512–2048 | High (>4 GB VRAM) | MIT License | Integrates textual reasoning with forecasting | Heavy runtime requirements; complex multi-task routing |
| **EXAONE-Forecast-Agent** | LG AI Research | Attention-Free 1D Conv + Group Pooling MLP | ~50M – 150M | ~200 – 600 MB | 512 / 96 | Medium | EXAONE Non-Commercial 1.2 | Linear-time operations; robust missing data handling | Non-commercial model license |
| **IBM Tiny Time Mixers (TTM)** | IBM Research | TSMixer-based Multilevel MLP Mixer | ~1M (0.8M – 1.5M) | ~4 MB | 512 / 96–192 | Minimal (<100 MB VRAM, CPU fast) | Apache 2.0 | SOTA zero-shot at 1M params; runs anywhere | Fixed context length and architecture depth |
| **PatchTST** | Nie et al. (ICLR 2023) | Channel-Independent Patch Transformer | ~200k – 600k | ~1 – 3 MB | 336–512 / 96–720 | Minimal (<200 MB VRAM) | Apache 2.0 | Subseries patching reduces token sequence length; strong SOTA baseline | Quadratic attention across patches; per-dataset training |
| **TSMixer** | Ekambaram et al. (2023) | Residual Cross-Time & Cross-Feature MLPs | ~100k – 800k | ~1 – 4 MB | 96–512 / 96–720 | Minimal (<150 MB VRAM) | Apache 2.0 | Pure MLP architecture; fast matrix multiplication | Requires fixed variate count in channel-dependent mode |
| **DLinear / NLinear** | Zeng et al. (AAAI 2023) | Single-Layer Linear with Decomposition / Normalization | ~2k – 20k | ~20 – 100 KB | 96–336 / 96–720 | Ultra-low (<50 MB VRAM, microseconds on CPU) | MIT / Apache 2.0 | Ultra-fast; strong baseline on linear trend datasets | Underfits multi-frequency harmonics; lacks cross-series transfer |
| **AutoETS & Seasonal Naive** | Classical Statistics | State-space Exponential Smoothing / Seasonal lag | 0 to 4 parameters | In-memory only | Full history / Horizon | Negligible (CPU only) | Public Domain / Open Source | Instantaneous; strong baseline on clean periodic series | Cannot capture cross-series features or nonlinear dynamics |

---

## 4. The CompactTS Architectural Hypothesis

Based on our survey of the empirical Pareto frontier, we formulate the **CompactTS Hypothesis**:

> **Hypothesis:** By combining Reversible Instance Normalization (RevIN) with hierarchical multi-scale moving-average decomposition, multi-kernel depthwise dilated temporal convolutions, and gated residual temporal mixing (GLU), a compact model with **100k to 1M parameters** can capture both low-frequency macro trends and high-frequency seasonal harmonics, delivering forecasting accuracy competitive with foundation models that are 100x to 500x larger, while requiring under 200 MB VRAM and achieving single-digit millisecond inference latency on consumer laptops and CPUs.

### Design Principles:
1. **Reversible Instance Normalization (RevIN):** Eliminates distribution shift between training windows and evaluation horizons without leaking future statistics.
2. **Multi-Scale Hierarchical Decomposition:** Decouples the smooth macro trend from high-frequency oscillations. The trend is projected via a direct linear operator, while the oscillatory residual is processed by nonlinear gated blocks.
3. **Multi-Kernel Dilated Convolutions:** Dilated depthwise 1D convolutions ($k \in \{3, 7, 15\}$, $d \in \{1, 2, 4\}$) capture multiple periodicities (hourly, daily, weekly) with negligible parameter overhead ($O(K \cdot C)$).
4. **Channel Independence (CI) by Default:** Processing each time series channel independently prevents overfitting to cross-channel spurious correlations and enables universal zero-shot transfer across datasets of arbitrary dimension.
5. **Calibrated Multi-Head Output:** Native support for both point prediction (Huber loss) and multi-quantile probabilistic prediction (Pinball loss for CRPS calibration).
