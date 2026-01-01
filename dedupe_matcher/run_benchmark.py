#!/usr/bin/env python3
"""Standalone dedupe benchmark runner."""

import json
import sys
import time

import polars as pl

from matcher import dedupe_match


def run_benchmark(data_path: str, threshold: float) -> dict:
    """Run dedupe benchmark on provided data."""
    data = json.loads(open(data_path).read())

    df_left = pl.DataFrame(data["left"])
    df_right = pl.DataFrame(data["right"])

    fields = data["fields"]
    weights = data["weights"]

    start = time.perf_counter()

    result_df = dedupe_match(
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
        "method": "dedupe",
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_benchmark.py <data_path> <threshold>")
        sys.exit(1)

    data_path = sys.argv[1]
    threshold = float(sys.argv[2])

    result = run_benchmark(data_path, threshold)
    print(json.dumps(result))
