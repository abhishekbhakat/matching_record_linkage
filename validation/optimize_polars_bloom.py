#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Quick optimizer for polars_bloom_jaccard only."""

import json
import subprocess
import tempfile
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

GOLDEN_RATIO = (5**0.5 - 1) / 2


def load_ground_truth() -> set[tuple[int, int]]:
    truth = pl.read_csv(SCRIPT_DIR / "ground_truth.csv")
    return set(
        (row["left_id"], row["right_id"])
        for row in truth.filter(pl.col("is_match") == 1).iter_rows(named=True)
    )


def load_validation_data() -> dict:
    left = pl.read_csv(SCRIPT_DIR / "source_left.csv")
    right = pl.read_csv(SCRIPT_DIR / "source_right.csv")
    return {
        "left": left.rename({"id": "left_id"}).to_dicts(),
        "right": right.rename({"id": "right_id"}).to_dicts(),
        "fields": ["name"],
        "weights": {"name": 1.0},
    }


def get_predictions(threshold: float, data_path: str) -> set[tuple[int, int]]:
    venv_path = PROJECT_DIR / ".venvs" / "polars_bloom_jaccard"
    script_path = PROJECT_DIR / "polars_bloom_jaccard"

    code = f'''
import json
import sys
sys.path.insert(0, "{script_path}")
import polars as pl
from matcher_native import polars_bloom_match

with open("{data_path}") as f:
    data = json.load(f)

df_left = pl.DataFrame(data["left"])
df_right = pl.DataFrame(data["right"])
result = polars_bloom_match(df_left, df_right, data["fields"], data["weights"], {threshold})
pairs = [(int(r["left_idx"]), int(r["right_idx"])) for r in result.iter_rows(named=True)]
print(json.dumps(pairs))
'''

    result = subprocess.run(
        [str(venv_path / "bin" / "python"), "-c", code],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr[:300]}")
        return set()

    try:
        pairs = json.loads(result.stdout.strip())
        return set(tuple(p) for p in pairs)
    except (json.JSONDecodeError, ValueError):
        return set()


def compute_f1(predicted: set, truth: set) -> tuple[float, float, float]:
    if not predicted:
        return 0.0, 0.0, 0.0

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def golden_section_search(truth: set, data_path: str, low: float = 0.1, high: float = 0.95, tol: float = 0.02) -> tuple[float, list[dict]]:
    history = []

    c = high - GOLDEN_RATIO * (high - low)
    d = low + GOLDEN_RATIO * (high - low)

    pred_c = get_predictions(c, data_path)
    prec_c, rec_c, f1_c = compute_f1(pred_c, truth)
    history.append({"threshold": round(c, 4), "f1": round(f1_c, 4), "precision": round(prec_c, 4), "recall": round(rec_c, 4), "matches": len(pred_c)})
    print(f"  t={c:.4f} -> F1={f1_c:.4f}, P={prec_c:.4f}, R={rec_c:.4f}, matches={len(pred_c)}")

    pred_d = get_predictions(d, data_path)
    prec_d, rec_d, f1_d = compute_f1(pred_d, truth)
    history.append({"threshold": round(d, 4), "f1": round(f1_d, 4), "precision": round(prec_d, 4), "recall": round(rec_d, 4), "matches": len(pred_d)})
    print(f"  t={d:.4f} -> F1={f1_d:.4f}, P={prec_d:.4f}, R={rec_d:.4f}, matches={len(pred_d)}")

    iterations = 0
    max_iterations = 12

    while abs(high - low) > tol and iterations < max_iterations:
        iterations += 1

        if f1_c > f1_d:
            high = d
            d = c
            f1_d = f1_c
            c = high - GOLDEN_RATIO * (high - low)
            pred_c = get_predictions(c, data_path)
            prec_c, rec_c, f1_c = compute_f1(pred_c, truth)
            history.append({"threshold": round(c, 4), "f1": round(f1_c, 4), "precision": round(prec_c, 4), "recall": round(rec_c, 4), "matches": len(pred_c)})
            print(f"  t={c:.4f} -> F1={f1_c:.4f}, P={prec_c:.4f}, R={rec_c:.4f}, matches={len(pred_c)}")
        else:
            low = c
            c = d
            f1_c = f1_d
            d = low + GOLDEN_RATIO * (high - low)
            pred_d = get_predictions(d, data_path)
            prec_d, rec_d, f1_d = compute_f1(pred_d, truth)
            history.append({"threshold": round(d, 4), "f1": round(f1_d, 4), "precision": round(prec_d, 4), "recall": round(rec_d, 4), "matches": len(pred_d)})
            print(f"  t={d:.4f} -> F1={f1_d:.4f}, P={prec_d:.4f}, R={rec_d:.4f}, matches={len(pred_d)}")

    optimal = (low + high) / 2
    pred_opt = get_predictions(optimal, data_path)
    prec_opt, rec_opt, f1_opt = compute_f1(pred_opt, truth)
    history.append({"threshold": round(optimal, 4), "f1": round(f1_opt, 4), "precision": round(prec_opt, 4), "recall": round(rec_opt, 4), "matches": len(pred_opt), "optimal": True})
    
    return optimal, history


def main():
    print("=" * 60)
    print("Polars Bloom Jaccard Threshold Optimizer")
    print("Type-aware ngram encoding: strings=2, dates=3, integers=3")
    print("=" * 60)

    truth = load_ground_truth()
    data = load_validation_data()
    print(f"\nGround truth: {len(truth)} true matches")
    print(f"Dataset: {len(data['left'])} x {len(data['right'])} = {len(data['left']) * len(data['right'])} comparisons")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_path = f.name

    print("\nRunning golden section search...\n")
    optimal, history = golden_section_search(truth, data_path)

    Path(data_path).unlink()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    best = max(history, key=lambda x: x["f1"])
    print(f"\nOptimal threshold: {best['threshold']:.4f}")
    print(f"F1 Score: {best['f1']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"Matches: {best['matches']}")


if __name__ == "__main__":
    main()
