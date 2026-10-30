# Commercial Usability & Licensing Audit

**Project:** CompactTS  
**Date:** October 20, 2026  
**Status:** Initial Phase Audit Complete

This document tracks all external components, code dependencies, pretrained foundation model checkpoints, and evaluation datasets examined or integrated within CompactTS, auditing their intellectual property terms and suitability for commercial downstream applications.

---

## 1. Executive Summary & Policy Guidelines

1. **Dual-License Caution in Foundation Models:**  
   Open-source code releases frequently differ in licensing from their corresponding pretrained weights. Several recent models (e.g., Timestral, TimesFM 3.0, EXAONE) provide open inference code under permissive licenses (Apache 2.0 / MIT) while restricting weight downloads to non-commercial research or revenue-capped tiers.
2. **CompactTS Licensing Stance:**  
   - All newly developed CompactTS code is released under the permissive **Business Source License 1.1 (BSL-1.1)**.
   - No proprietary or non-commercial weights are embedded into CompactTS checkpoints.
   - For benchmark reproduction, zero-shot foundation models are categorized into **Commercially Unrestricted**, **Revenue-Capped**, and **Research-Only (Non-Commercial)**.

---

## 2. Licensing Audit Table

| Component Name | Type | Source URL / Repository | License Identifier | Intended Project Use | Commercial Use Status | Specific Restrictions & Notes | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CompactTS** | Software & Weights | `https://github.com/DakshBhalala/CompactTS-Agent` | **BSL-1.1** | Core project codebase & models | **PERMITTED** | Full commercial and redistribution rights granted under Apache 2.0. | Verified |
| **PyTorch** | Dependency | `https://github.com/pytorch/pytorch` | **Modified BSD-3-Clause** | Core deep learning runtime | **PERMITTED** | Standard permissive software license. | Verified |
| **NumPy & SciPy** | Dependency | `https://numpy.org`, `https://scipy.org` | **BSD-3-Clause** | Numerical array & optimization ops | **PERMITTED** | Permissive open source. | Verified |
| **Pandas** | Dependency | `https://pandas.pydata.org` | **BSD-3-Clause** | Time-series data manipulation | **PERMITTED** | Permissive open source. | Verified |
| **scikit-learn** | Dependency | `https://scikit-learn.org` | **BSD-3-Clause** | Preprocessing, metrics, ridge baseline | **PERMITTED** | Permissive open source. | Verified |
| **LightGBM** | Dependency | `https://github.com/microsoft/LightGBM` | **MIT License** | Gradient-boosted decision tree baseline | **PERMITTED** | Permissive open source. | Verified |
| **einops** | Dependency | `https://github.com/arogozhnikov/einops` | **MIT License** | Tensor dimensions reshaping | **PERMITTED** | Permissive open source. | Verified |
| **GIFT-Eval Benchmark Code** | Benchmark Suite | `https://github.com/SalesforceAIResearch/gift-eval` | **Apache-2.0** | Evaluation protocols & metric schemas | **PERMITTED** | Evaluation code and scripts are permissively licensed. | Verified |
| **GiftEvalPretrain** | Dataset | `https://huggingface.co/datasets/Salesforce/GiftEval` | **Multiple / Open Data** | Pretraining corpus | **CONDITIONAL REVIEW** | Aggregation of diverse open-access time-series datasets. Individual domain sets may possess distinct CC licenses (CC-BY, CC0, etc.). | Requires Dataset-Level Audit |
| **Chronos & Chronos-2** | Model Weights & Code | `https://github.com/amazon-science/chronos-forecasting` | **Apache-2.0** | Baseline foundation model comparison | **PERMITTED** | Amazon Science released code and model checkpoints under Apache 2.0. | Verified |
| **TimesFM 1.0 / 2.5** | Model Weights & Code | `https://github.com/google-research/timesfm` | **Apache-2.0** | Baseline foundation model comparison | **PERMITTED** | Code and weights up to version 2.5 are Apache 2.0 licensed. | Verified |
| **TimesFM 3.0** | Model Weights | `https://huggingface.co/google/timesfm-3.0` | **Non-Commercial Research** | Research comparison only | **RESTRICTED** | Weights carry non-commercial research terms; commercial access requires Google Cloud BigQuery API. | Verified |
| **Timestral-209M / 5M** | Model Code | `https://github.com/mistralai/timestral` | **Apache-2.0** | Standalone inference reference | **PERMITTED** | Standalone codebase is Apache 2.0. | Verified |
| **Timestral Weights** | Pretrained Weights | `https://huggingface.co/mistralai/Timestral-209M` | **Modified MIT** | Baseline zero-shot comparison | **RESTRICTED (Revenue Cap)** | Strict commercial revenue cap: entities with >$20M USD consolidated monthly revenue cannot use without paid agreement. | Verified |
| **TiRex-2** | Model Weights & Code | `https://github.com/NX-AI` | **Apache-2.0** | Recurrent foundation model baseline | **PERMITTED** | TiRex-2 code and checkpoints released under Apache 2.0 (TiRex 1.0 was under restrictive NXAI Community license). | Verified |
| **Moirai / Uni2TS** | Library & Checkpoints | `https://github.com/SalesforceAIResearch/uni2ts` | **Apache-2.0** | Universal transformer baseline | **PERMITTED** | Salesforce Uni2TS codebase and Moirai-1.0-R checkpoints are Apache 2.0. | Verified |
| **EXAONE-Forecast-Agent** | Model Weights | `https://huggingface.co/LGAI-EXAONE` | **EXAONE AI Model License 1.2 - NC** | Research comparison | **PROHIBITED COMMERCIALLY** | Model weights explicitly restricted to non-commercial academic research. | Verified |
| **OpenTSLM** | Model & Code | `https://github.com/OpenTSLM` | **MIT License** | TSLM baseline comparison | **PERMITTED** | Permissive open source. | Verified |
| **TW3Cast** | Router & Config | `https://github.com/TW3-Partners-OS/TW3-Cast` | **Open Access / arXiv** | Routing tournament study | **PERMITTED** | Routing table and evaluation scripts open source. | Verified |
| **IBM Tiny Time Mixers (TTM)** | Model Weights & Code | `https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1` | **Apache-2.0** | Compact foundation model comparison | **PERMITTED** | Part of IBM Granite, open for commercial use under Apache 2.0. | Verified |

---

## 3. Commercial Usability Guidelines for CompactTS

1. **Clean Room Implementation:**  
   CompactTS is written entirely from first principles using standard PyTorch, NumPy, SciPy, and scikit-learn primitives. It does not incorporate or sub-license any proprietary code, third-party neural network weights, or non-commercial research models.
2. **Safe Pretraining Datasets:**  
   Any pretraining data utilized for CompactTS must be screened for:
   - Permissive licenses (CC0, CC-BY, MIT, Apache-2.0).
   - Datasets explicitly prohibiting commercial distribution (e.g., CC-BY-NC) must be isolated or excluded from production pretraining pipelines.
3. **Distribution Guidance:**  
   CompactTS checkpoints trained using synthetic generator scripts or verified open datasets are 100% free of licensing encumbrances and may be deployed, packaged, or monetized commercially.
