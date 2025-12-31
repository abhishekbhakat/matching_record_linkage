#!/usr/bin/env python3
"""Standalone Splink benchmark runner."""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["SPLINK_QUIET_MODE"] = "1"

import polars as pl

from matcher import splink_match


def run_benchmark(data_path: str, threshold: float) -> dict:
    """Run Splink benchmark on provided data."""
    with open(data_path) as f:
        data = json.load(f)

    df_left = pl.DataFrame(data["left"])
    df_right = pl.DataFrame(data["right"])

    fields = data["fields"]
    weights = data["weights"]

    start = time.perf_counter()

    try:
        result_df = splink_match(
            df_left=df_left,
            df_right=df_right,
            fields=fields,
            weights=weights,
            threshold=threshold,
        )
        elapsed = time.perf_counter() - start
        return {
            "time": elapsed,
            "matches": len(result_df),
            "method": "splink",
        }
    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - start
        return {
            "time": elapsed,
            "matches": 0,
            "method": "splink",
            "error": traceback.format_exc(),
        }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <data_path> <threshold>")
        sys.exit(1)

    data_path = sys.argv[1]
    threshold = float(sys.argv[2])

    result = run_benchmark(data_path, threshold)
    print(json.dumps(result))
