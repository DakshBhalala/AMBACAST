# Pull Request: Add CompactTS-Agent Results

**Target Repository:** `SalesforceAIResearch/gift-eval`  
**Base Branch:** `main`  
**PR Title:** `[Submission] Add CompactTS-Agent results`

---

## Submission Details

- **Model Name:** `CompactTS-Agent`
- **Model Type:** `agentic`
- **Model Data Type:** `float32`
- **Organization / Author:** `Daksh Bhalala`
- **Test Data Leakage:** `No`
- **Replication Code Available:** `Yes`
- **Code Repository:** https://github.com/DakshBhalala/AMBACAST
- **Model / Checkpoint Link:** https://huggingface.co/BhalalaDaksh/CompactTS-Agent

---

## Description

This submission adds evaluation results for **CompactTS-Agent**, an autonomous agentic meta-forecasting system. 

### Method Overview
CompactTS-Agent extracts structural statistical and spectral properties from input time series (including Fourier power spectrum, autocorrelation period, trend variance ratio, and kurtosis) and dynamically routes forecasting tasks across domain-specialized open-weights foundation model architectures (`TimesFM-3`, `Chronos-2`, `Toto-2.0`). All quantiles ($q_{0.1}$ through $q_{0.9}$) are calibrated to enforce strict monotonicity.

All underlying foundation models utilized in this agentic framework are public open-source weights under Apache-2.0 / MIT licenses.

---

## Validation Checklist

- [x] Submission folder named `results/CompactTS-Agent` matches `config.json` model name.
- [x] Directory contains strictly `all_results.csv` and `config.json`.
- [x] `all_results.csv` contains all 97 benchmark configurations in exact official order.
- [x] All 15 required metric columns are present and contain valid numerical values.
- [x] Passed official validation script:
  ```bash
  python scripts/validate_results.py results/CompactTS-Agent
  # PASS results\CompactTS-Agent
  ```
- [x] Evaluation replication scripts are publicly accessible at https://github.com/DakshBhalala/AMBACAST.
