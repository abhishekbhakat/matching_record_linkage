#!/usr/bin/env python3
"""Evaluate matchers against ground truth to find optimal threshold."""

import sys
from pathlib import Path

import polars as pl
import yaml

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent


def load_data():
    """Load source CSVs and ground truth."""
    left = pl.read_csv(SCRIPT_DIR / "source_left.csv")
    right = pl.read_csv(SCRIPT_DIR / "source_right.csv")
    truth = pl.read_csv(SCRIPT_DIR / "ground_truth.csv")
    return left, right, truth


def load_config(matcher_name: str) -> dict:
    """Load config.yaml for a matcher."""
    config_path = ROOT_DIR / matcher_name / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"thresholds": {"search_range": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}}


def get_polars_bloom_scores(df_left: pl.DataFrame, df_right: pl.DataFrame) -> pl.DataFrame:
    """Get scores using polars_bloom_jaccard matcher."""
    sys.path.insert(0, str(ROOT_DIR / "polars_bloom_jaccard"))
    from matcher_fast import polars_bloom_match_fast
    return polars_bloom_match_fast(
        df_left=df_left, df_right=df_right,
        fields=["name"], weights={"name": 1.0}, threshold=0.0,
    )


def evaluate_threshold(scores: pl.DataFrame, truth: pl.DataFrame, threshold: float) -> dict:
    """Evaluate precision/recall at a given threshold."""
    predicted = scores.filter(pl.col("score") >= threshold)
    
    true_matches = set(
        (row["left_id"], row["right_id"])
        for row in truth.filter(pl.col("is_match") == 1).iter_rows(named=True)
    )
    
    predicted_matches = set(
        (row["left_idx"], row["right_idx"])
        for row in predicted.iter_rows(named=True)
    )
    
    tp = len(true_matches & predicted_matches)
    fp = len(predicted_matches - true_matches)
    fn = len(true_matches - predicted_matches)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_evaluation(matcher_name: str, scores: pl.DataFrame, truth: pl.DataFrame, config: dict):
    """Run threshold evaluation for a single matcher."""
    thresholds = config.get("thresholds", {}).get("search_range", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    print(f"\n{'='*70}")
    print(f"THRESHOLD EVALUATION: {matcher_name}")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    best_f1, best_threshold = 0.0, 0.0
    results = []
    
    for t in thresholds:
        result = evaluate_threshold(scores, truth, t)
        results.append(result)
        print(f"{result['threshold']:<12.2f} {result['tp']:<6} {result['fp']:<6} {result['fn']:<6} "
              f"{result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_threshold = t
    
    print("-" * 70)
    print(f"BEST: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    return {"matcher": matcher_name, "best_threshold": best_threshold, "best_f1": best_f1, "results": results}


def main():
    print("Loading data...")
    left, right, truth = load_data()
    
    print(f"Left: {len(left)} rows, Right: {len(right)} rows")
    true_match_count = truth.filter(pl.col("is_match") == 1).height
    print(f"Ground truth: {len(truth)} pairs ({true_match_count} true matches, {len(truth) - true_match_count} non-matches)")
    
    summary = []
    
    print("\n" + "=" * 70)
    print("POLARS BLOOM JACCARD")
    print("=" * 70)
    config = load_config("polars_bloom_jaccard")
    print("Computing scores...")
    scores = get_polars_bloom_scores(left, right)
    print(f"Pairs scored: {len(scores)}")
    result = run_evaluation("polars_bloom_jaccard", scores, truth, config)
    summary.append(result)
    
    print("\n\n" + "=" * 70)
    print("SUMMARY: OPTIMAL THRESHOLDS")
    print("=" * 70)
    for r in summary:
        print(f"{r['matcher']:<30} threshold={r['best_threshold']:.2f}  F1={r['best_f1']:.4f}")


if __name__ == "__main__":
    main()
