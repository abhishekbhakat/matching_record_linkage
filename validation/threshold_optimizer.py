#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Smart threshold optimizer using golden section search.

Finds optimal threshold for each matcher by maximizing F1 score.
Uses golden section search instead of brute force.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

MATCHERS = [
    ("clkhash_matcher", "CLKHash"),
    ("polars_bloom_jaccard", "Polars Bloom"),
    ("modular_pprl", "Modular PPRL"),
    ("rapidfuzz_matcher", "RapidFuzz"),
    ("modernbert_matcher", "ModernBERT"),
]

GOLDEN_RATIO = (5**0.5 - 1) / 2


def load_ground_truth() -> set[tuple[int, int]]:
    """Load ground truth match pairs."""
    truth = pl.read_csv(SCRIPT_DIR / "ground_truth.csv")
    return set(
        (row["left_id"], row["right_id"])
        for row in truth.filter(pl.col("is_match") == 1).iter_rows(named=True)
    )


def load_validation_data() -> dict:
    """Load validation dataset."""
    left = pl.read_csv(SCRIPT_DIR / "source_left.csv")
    right = pl.read_csv(SCRIPT_DIR / "source_right.csv")
    return {
        "left": left.rename({"id": "left_id"}).to_dicts(),
        "right": right.rename({"id": "right_id"}).to_dicts(),
        "fields": ["name"],
        "weights": {"name": 1.0},
    }


def run_matcher(matcher_dir: str, data_path: str, threshold: float) -> set[tuple[int, int]]:
    """Run a matcher and return predicted match pairs."""
    venv_path = PROJECT_DIR / ".venvs" / matcher_dir
    script_path = PROJECT_DIR / matcher_dir / "run_benchmark.py"

    if not venv_path.exists() or not script_path.exists():
        return set()

    result = subprocess.run(
        [str(venv_path / "bin" / "python"), str(script_path), data_path, str(threshold)],
        capture_output=True,
        text=True,
        cwd=script_path.parent,
    )

    if result.returncode != 0:
        return set()

    try:
        output = json.loads(result.stdout.strip())
        return set()
    except json.JSONDecodeError:
        return set()


def get_predictions_from_matcher(matcher_dir: str, threshold: float, data_path: str) -> set[tuple[int, int]]:
    """Get predicted pairs from matcher at given threshold."""
    venv_path = PROJECT_DIR / ".venvs" / matcher_dir
    script_path = PROJECT_DIR / matcher_dir

    if not venv_path.exists():
        return set()

    code = f'''
import json
import sys
sys.path.insert(0, "{script_path}")
import polars as pl

with open("{data_path}") as f:
    data = json.load(f)

df_left = pl.DataFrame(data["left"])
df_right = pl.DataFrame(data["right"])

'''
    if matcher_dir == "clkhash_matcher":
        code += f'''
from matcher import pprl_fuzzy_match
result = pprl_fuzzy_match(df_left, df_right, data["fields"], data["weights"], {threshold})
'''
    elif matcher_dir == "polars_bloom_jaccard":
        code += f'''
from matcher import polars_bloom_match
result = polars_bloom_match(df_left, df_right, data["fields"], data["weights"], {threshold})
'''
    elif matcher_dir == "modular_pprl":
        code += f'''
from matcher import modular_pprl_match
result = modular_pprl_match(df_left, df_right, data["fields"], data["weights"], {threshold})
'''
    elif matcher_dir == "rapidfuzz_matcher":
        code += f'''
from matcher import rapidfuzz_match
result = rapidfuzz_match(df_left, df_right, data["fields"], data["weights"], {threshold})
'''
    elif matcher_dir == "modernbert_matcher":
        code += f'''
from matcher import modernbert_match
result = modernbert_match(df_left, df_right, data["fields"], data["weights"], {threshold})
'''
    else:
        return set()

    code += '''
pairs = [(int(r["left_idx"]), int(r["right_idx"])) for r in result.iter_rows(named=True)]
print(json.dumps(pairs))
'''

    result = subprocess.run(
        [str(venv_path / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
        return set()

    try:
        pairs = json.loads(result.stdout.strip())
        return set(tuple(p) for p in pairs)
    except (json.JSONDecodeError, ValueError):
        return set()


def compute_f1(predicted: set, truth: set) -> tuple[float, float, float]:
    """Compute precision, recall, F1."""
    if not predicted:
        return 0.0, 0.0, 0.0

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def golden_section_search(
    matcher_dir: str,
    truth: set,
    data_path: str,
    low: float = 0.1,
    high: float = 0.95,
    tol: float = 0.02,
) -> tuple[float, list[dict]]:
    """Find optimal threshold using golden section search."""
    history = []

    c = high - GOLDEN_RATIO * (high - low)
    d = low + GOLDEN_RATIO * (high - low)

    pred_c = get_predictions_from_matcher(matcher_dir, c, data_path)
    _, _, f1_c = compute_f1(pred_c, truth)
    history.append({"threshold": round(c, 4), "f1": round(f1_c, 4), "matches": len(pred_c)})

    pred_d = get_predictions_from_matcher(matcher_dir, d, data_path)
    _, _, f1_d = compute_f1(pred_d, truth)
    history.append({"threshold": round(d, 4), "f1": round(f1_d, 4), "matches": len(pred_d)})

    iterations = 0
    max_iterations = 15

    while abs(high - low) > tol and iterations < max_iterations:
        iterations += 1

        if f1_c > f1_d:
            high = d
            d = c
            f1_d = f1_c
            c = high - GOLDEN_RATIO * (high - low)
            pred_c = get_predictions_from_matcher(matcher_dir, c, data_path)
            _, _, f1_c = compute_f1(pred_c, truth)
            history.append({"threshold": round(c, 4), "f1": round(f1_c, 4), "matches": len(pred_c)})
        else:
            low = c
            c = d
            f1_c = f1_d
            d = low + GOLDEN_RATIO * (high - low)
            pred_d = get_predictions_from_matcher(matcher_dir, d, data_path)
            _, _, f1_d = compute_f1(pred_d, truth)
            history.append({"threshold": round(d, 4), "f1": round(f1_d, 4), "matches": len(pred_d)})

    optimal_threshold = (low + high) / 2

    pred_opt = get_predictions_from_matcher(matcher_dir, optimal_threshold, data_path)
    prec, rec, f1_opt = compute_f1(pred_opt, truth)
    history.append({
        "threshold": round(optimal_threshold, 4),
        "f1": round(f1_opt, 4),
        "matches": len(pred_opt),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "optimal": True,
    })

    return optimal_threshold, history


def main():
    print("Loading ground truth and validation data...")
    truth = load_ground_truth()
    data = load_validation_data()

    print(f"Ground truth: {len(truth)} true matches")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_path = f.name

    all_results = []
    optimal_thresholds = {}

    for matcher_dir, matcher_name in MATCHERS:
        print(f"\n{'='*60}")
        print(f"Optimizing: {matcher_name}")
        print("=" * 60)

        venv_path = PROJECT_DIR / ".venvs" / matcher_dir
        if not venv_path.exists():
            print(f"  Skipping (venv not found)")
            continue

        optimal, history = golden_section_search(matcher_dir, truth, data_path)
        optimal_thresholds[matcher_name] = optimal

        for h in history:
            h["matcher"] = matcher_name
            all_results.append(h)

        print(f"\nSearch history:")
        for h in sorted(history, key=lambda x: x["threshold"]):
            opt_mark = " *OPTIMAL*" if h.get("optimal") else ""
            print(f"  t={h['threshold']:.2f} -> F1={h['f1']:.4f}, matches={h['matches']}{opt_mark}")

        best = max(history, key=lambda x: x["f1"])
        print(f"\nBest: threshold={best['threshold']:.4f}, F1={best['f1']:.4f}")

    Path(data_path).unlink()

    results_df = pl.DataFrame(all_results)
    results_df.write_csv(SCRIPT_DIR / "threshold_optimization_results.csv")

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLDS SUMMARY")
    print("=" * 60)
    for matcher, threshold in optimal_thresholds.items():
        best = [r for r in all_results if r["matcher"] == matcher and r.get("optimal")]
        if best:
            b = best[0]
            print(f"{matcher:20} -> threshold={threshold:.4f}, F1={b['f1']:.4f}, P={b.get('precision', 0):.4f}, R={b.get('recall', 0):.4f}")

    print(f"\nResults saved to: validation/threshold_optimization_results.csv")
    print("Use this CSV for plotting threshold vs F1 curves.")


if __name__ == "__main__":
    main()
