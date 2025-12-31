#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pyyaml"]
# ///
"""Comprehensive benchmark runner for PPRL implementations.

Compares:
- CLKHash (legacy, archived 2023)
- Polars Bloom (our CLKHash-compatible impl)
- blooms library (2025 replacement)
- recordlinkage (stable Python toolkit)
- Splink (industry standard, non-PPRL)
"""

import json
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import yaml

SCRIPT_DIR = Path(__file__).parent

IMPLEMENTATIONS = [
    ("clkhash_matcher", "CLKHash (legacy)"),
    ("polars_bloom_jaccard", "Polars Bloom (ours)"),
    ("modular_pprl", "Modular PPRL (blooms+rl)"),
    ("splink_matcher", "Splink (unsupervised)"),
    ("rapidfuzz_matcher", "RapidFuzz (token_set)"),
]

VALIDATION_DIR = SCRIPT_DIR / "validation"
DEFAULT_THRESHOLD = 0.5


def load_matcher_config(impl_dir: str) -> dict:
    """Load matcher's config.yaml if it exists."""
    config_path = SCRIPT_DIR / impl_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_threshold(impl_dir: str) -> float:
    """Get optimal threshold from matcher's config or use default."""
    config = load_matcher_config(impl_dir)
    return config.get("thresholds", {}).get("default", DEFAULT_THRESHOLD)


def load_validation_data() -> dict:
    """Load the curated validation dataset from CSVs."""
    left = pl.read_csv(VALIDATION_DIR / "source_left.csv")
    right = pl.read_csv(VALIDATION_DIR / "source_right.csv")
    
    df_left = left.rename({"id": "left_id"})
    df_right = right.rename({"id": "right_id"})
    
    return {
        "left": df_left.to_dicts(),
        "right": df_right.to_dicts(),
        "fields": ["name"],
        "weights": {"name": 1.0},
    }


def load_ground_truth() -> set[tuple[int, int]]:
    """Load ground truth match pairs (by index position)."""
    truth = pl.read_csv(VALIDATION_DIR / "ground_truth.csv")
    return set(
        (row["left_id"], row["right_id"])
        for row in truth.filter(pl.col("is_match") == 1).iter_rows(named=True)
    )


def setup_venv(impl_dir: Path) -> Path:
    """Create isolated venv using uv."""
    venv_path = SCRIPT_DIR / ".venvs" / impl_dir.name
    req_file = impl_dir / "requirements.txt"

    if not venv_path.exists():
        print(f"  Creating venv: {venv_path.name}")
        subprocess.run(
            ["uv", "venv", str(venv_path)],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["uv", "pip", "install", "-p", str(venv_path), "-r", str(req_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Install error: {result.stderr}")
            raise RuntimeError(f"Failed to install deps: {result.stderr}")

    return venv_path


def run_in_venv(venv_path: Path, script_path: Path, data_path: str, threshold: float) -> dict:
    """Run benchmark script in isolated venv."""
    python_path = venv_path / "bin" / "python"

    result = subprocess.run(
        [str(python_path), str(script_path), data_path, str(threshold)],
        capture_output=True,
        text=True,
        cwd=script_path.parent,
    )

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return {"time": 0, "matches": 0, "method": "error", "error": result.stderr}

    return json.loads(result.stdout.strip())


def get_predictions(venv_path: Path, impl_dir: str, data_path: str, threshold: float) -> set[tuple[int, int]]:
    """Run matcher and get predicted pairs for metrics calculation."""
    script_path = SCRIPT_DIR / impl_dir
    
    func_map = {
        "clkhash_matcher": "pprl_fuzzy_match",
        "polars_bloom_jaccard": "polars_bloom_match",
        "modular_pprl": "modular_pprl_match",
        "rapidfuzz_matcher": "rapidfuzz_match",
        "splink_matcher": "splink_match",
    }
    
    func_name = func_map.get(impl_dir)
    if not func_name:
        return set()
    
    code = f'''
import json
import sys
sys.path.insert(0, "{script_path}")
import polars as pl
from matcher import {func_name}

with open("{data_path}") as f:
    data = json.load(f)

df_left = pl.DataFrame(data["left"])
df_right = pl.DataFrame(data["right"])
result = {func_name}(df_left, df_right, data["fields"], data["weights"], {threshold})
pairs = [(int(r["left_idx"]), int(r["right_idx"])) for r in result.iter_rows(named=True)]
print(json.dumps(pairs))
'''
    
    result = subprocess.run(
        [str(venv_path / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        return set()
    
    try:
        pairs = json.loads(result.stdout.strip())
        return set(tuple(p) for p in pairs)
    except (json.JSONDecodeError, ValueError):
        return set()


def run_benchmark() -> pl.DataFrame:
    results = []
    ground_truth = load_ground_truth()
    print(f"\nGround truth: {len(ground_truth)} true matches")

    print("\n[Setup] Creating isolated virtual environments...")
    venvs = {}
    for impl_dir, impl_name in IMPLEMENTATIONS:
        impl_path = SCRIPT_DIR / impl_dir
        if impl_path.exists():
            try:
                venvs[impl_dir] = setup_venv(impl_path)
                print(f"  ✓ {impl_name}")
            except Exception as e:
                print(f"  ✗ {impl_name}: {e}")
    print("  Done.\n")

    test_data = load_validation_data()
    n = len(test_data["left"])
    
    print(f"{'='*70}")
    print(f"Validation dataset: {n} rows (cross-join = {n*n:,} comparisons)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        data_path = f.name

    row_result = {"rows": n, "comparisons": n * n}

    for impl_dir, impl_name in IMPLEMENTATIONS:
        if impl_dir not in venvs:
            continue

        impl_path = SCRIPT_DIR / impl_dir
        impl_threshold = get_threshold(impl_dir)
        print(f"\n[{impl_name}] Running (threshold={impl_threshold})...")

        result = run_in_venv(
            venvs[impl_dir],
            impl_path / "run_benchmark.py",
            data_path,
            impl_threshold,
        )

        time_val = result.get("time", 0)
        matches = result.get("matches", 0)
        error = result.get("error")

        predicted = get_predictions(venvs[impl_dir], impl_dir, data_path, impl_threshold)
        tp = len(predicted & ground_truth)
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if error:
            print(f"  ✗ Error: {error[:80]}...")
        else:
            print(f"  Time: {time_val:.4f}s | Matches: {matches}")
            print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        key = impl_dir.replace("_matcher", "").replace("_jaccard", "")
        row_result[f"{key}_time_s"] = round(time_val, 4)
        row_result[f"{key}_matches"] = matches
        row_result[f"{key}_threshold"] = impl_threshold
        row_result[f"{key}_precision"] = round(precision, 4)
        row_result[f"{key}_recall"] = round(recall, 4)
        row_result[f"{key}_f1"] = round(f1, 4)

    Path(data_path).unlink()
    results.append(row_result)

    return pl.DataFrame(results)


def print_summary(results: pl.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(results)

    time_cols = [c for c in results.columns if c.endswith("_time_s")]
    if len(time_cols) >= 2:
        print("\nAverage times:")
        for col in time_cols:
            name = col.replace("_time_s", "")
            avg = results[col].mean()
            if avg is not None:
                print(f"  • {name}: {float(avg):.4f}s")

    match_cols = [c for c in results.columns if c.endswith("_matches")]
    if match_cols:
        print("\nMatch counts (last row):")
        last_row = results.tail(1)
        for col in match_cols:
            name = col.replace("_matches", "")
            val = last_row[col].item()
            print(f"  • {name}: {val}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPRL benchmark using validation dataset")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("PPRL BENCHMARK (Validation Dataset)")
    print("=" * 70)
    print("\nImplementations (using config.yaml thresholds):")
    for impl_dir, impl_name in IMPLEMENTATIONS:
        threshold = get_threshold(impl_dir)
        print(f"  • {impl_name}: threshold={threshold}")
    print(f"\nDataset: validation/source_left.csv, source_right.csv")

    results = run_benchmark()
    print_summary(results)

    if args.output:
        Path(args.output).write_bytes(results.write_csv().encode())
        print(f"\nResults saved to {args.output}")
