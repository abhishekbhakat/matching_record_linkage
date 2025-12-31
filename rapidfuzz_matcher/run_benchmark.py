#!/usr/bin/env python3
"""Standalone RapidFuzz benchmark runner."""

import json
import sys
import time

import polars as pl

from matcher import rapidfuzz_match


def run_benchmark(data_path: str, threshold: float) -> dict:
    """Run RapidFuzz benchmark on provided data."""
    with open(data_path) as f:
        data = json.load(f)

    df_left = pl.DataFrame(data["left"])
    df_right = pl.DataFrame(data["right"])

    fields = data["fields"]
    weights = data["weights"]

    start = time.perf_counter()

    result_df = rapidfuzz_match(
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
        "method": "rapidfuzz",
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <data_path> <threshold>")
        sys.exit(1)

    data_path = sys.argv[1]
    threshold = float(sys.argv[2])

    result = run_benchmark(data_path, threshold)
    print(json.dumps(result))
