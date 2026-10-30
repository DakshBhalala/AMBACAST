"""Prepares and validates the official GIFT-Eval leaderboard submission directory.

Uses the verified CompactTS-Agent autonomous meta-forecaster routing across
open foundation models (Chronos-2, TimesFM-3, Toto-2.0).

Outputs:
1. results/CompactTS-Agent/config.json with required metadata fields and disclosures.
2. results/CompactTS-Agent/all_results.csv conforming to the exact 98-line, 15-column schema.
3. Automatically validates the generated submission using scripts/validate_results.py.
"""

from typing import Dict, Any
import os
import sys
from pathlib import Path

from compactts.agent.generate_submission import generate_agent_submission


def generate_submission(
    model_name: str = "CompactTS-Agent",
    output_dir: str = "results/CompactTS-Agent",
) -> None:
    """Generate and validate the GIFT-Eval leaderboard submission folder."""
    generate_agent_submission(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    generate_submission()
