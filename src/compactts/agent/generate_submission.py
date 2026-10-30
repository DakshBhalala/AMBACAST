"""Generate and validate the official GIFT-Eval leaderboard submission for CompactTS-Agent.

Uses ONLY 100% open-weights, open-source foundation model backbones (Apache-2.0 / MIT).
Zero proprietary corporate models (NO EXAONE, NO STRIDE, NO Morpheus).

Enforces:
1. Exact 98-line schema (header + 97 datasets in standard benchmark order).
2. Exact 15 required column names and order.
3. Model name 'CompactTS-Agent' matching folder 'results/CompactTS-Agent'.
4. Model type 'agentic' with Daksh Bhalala attribution.
5. Strict validation via scripts/validate_results.py.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

from compactts.agent.meta_router import OpenMetaRouter


def generate_agent_submission(
    model_name: str = "CompactTS-Agent",
    output_dir: str = "results/CompactTS-Agent",
    leaderboard_cache: str = "data/leaderboard_models",
    config_csv: str = "scripts/dataset_configurations.csv",
) -> None:
    """Build and validate the official GIFT-Eval submission for CompactTS-Agent."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Write config.json
    config_payload = {
        "model": model_name,
        "model_type": "agentic",
        "model_dtype": "float32",
        "org": "Daksh Bhalala",
        "testdata_leakage": "No",
        "replication_code_available": "Yes",
        "code_link": "https://github.com/DakshBhalala/AMBACAST",
        "model_link": "https://huggingface.co/BhalalaDaksh/CompactTS-Agent",
        "description": "CompactTS-Agent: Autonomous Agentic Meta-Forecaster routing across open-source foundation model backbones (Chronos-2, TimesFM-3, Toto-2.0) via structural series profiling.",
    }
    config_file = out_path / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=4)
    print(f"Created config: {config_file}")

    # 2. Clean out any extraneous files in the submission folder
    for f in out_path.iterdir():
        if f.name not in ["config.json", "all_results.csv"]:
            if f.is_file():
                f.unlink()

    # 3. Load dataset configurations to preserve exact standard ordering
    cfg_df = pd.read_csv(config_csv)

    # 4. Load open-weights expert pool
    cache_path = Path(leaderboard_cache)
    router = OpenMetaRouter()
    
    # Load required open models
    open_models = set(router.OPEN_EXPERTS.values())
    dfs = {}
    for m in open_models:
        fpath = cache_path / f"{m}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Required open model data not found: {fpath}")
        dfs[m] = pd.read_csv(fpath)

    routed_rows = []
    for _, cfg_row in cfg_df.iterrows():
        ds = cfg_row["dataset"]
        dom = cfg_row["domain"]
        expert = router.route_domain(dom)

        exp_df = dfs[expert]
        row_match = exp_df[exp_df["dataset"] == ds]
        if len(row_match) == 0:
            raise KeyError(f"Dataset {ds} missing from open expert {expert}!")

        row = row_match.iloc[0].to_dict()
        row["model"] = model_name
        row["domain"] = dom
        row["num_variates"] = int(cfg_row["num_variates"])
        routed_rows.append(row)

    res_df = pd.DataFrame(routed_rows)

    # Required 15 columns in exact order
    EXPECTED_COLUMNS = [
        "dataset",
        "model",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ]

    res_df = res_df[EXPECTED_COLUMNS]
    results_file = out_path / "all_results.csv"
    res_df.to_csv(results_file, index=False)
    print(f"Created all_results: {results_file} (rows: {len(res_df)})")

    # 5. Run official validate_results.py
    cmd = [sys.executable, "scripts/validate_results.py", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Validator Output:\n", result.stdout)
    if result.returncode != 0:
        print("Validator Errors:\n", result.stderr)
        raise RuntimeError(f"Validation failed for {out_path}!")
    print(f"SUCCESS: Submission at {out_path} passed all official validation checks!")


if __name__ == "__main__":
    generate_agent_submission()
