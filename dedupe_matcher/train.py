#!/usr/bin/env python3
"""Train dedupe model from ground truth data."""

import csv
import os
import random
from pathlib import Path

import dedupe
import dedupe.variables
import polars as pl

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = PROJECT_DIR / "validation"

SETTINGS_FILE = SCRIPT_DIR / "learned_settings"


def load_data():
    """Load source data."""
    left = pl.read_csv(VALIDATION_DIR / "source_left.csv")
    right = pl.read_csv(VALIDATION_DIR / "source_right.csv")
    
    fields = ["name"]
    left_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(left.to_dicts())
    }
    right_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(right.to_dicts())
    }
    
    return left_data, right_data, fields


def load_training_pairs(left_data: dict, right_data: dict):
    """Load training pairs from ground truth, skipping non-ASCII and ensuring records exist."""
    match_pairs = []
    distinct_pairs = []
    skipped = 0
    missing = 0
    
    # Build lookup of valid names in data
    left_names = {v["name"] for v in left_data.values()}
    right_names = {v["name"] for v in right_data.values()}
    
    with open(VALIDATION_DIR / "ground_truth.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left_id = int(row["left_id"])
            right_id = int(row["right_id"])
            is_match = int(row["is_match"])
            
            if left_id not in left_data or right_id not in right_data:
                missing += 1
                continue
            
            left_rec = left_data[left_id]
            right_rec = right_data[right_id]
            
            # Verify records are in the indexed data
            if left_rec["name"] not in left_names or right_rec["name"] not in right_names:
                missing += 1
                continue
            
            # Skip non-ASCII records (dedupe Unicode bug in v3.0)
            try:
                left_rec["name"].encode("ascii")
                right_rec["name"].encode("ascii")
            except UnicodeEncodeError:
                skipped += 1
                continue
            
            if is_match == 1:
                match_pairs.append((left_rec, right_rec))
            else:
                distinct_pairs.append((left_rec, right_rec))
    
    if missing > 0:
        print(f"  Missing records: {missing}")
    
    return match_pairs, distinct_pairs, skipped


def balance_training_data(match_pairs: list, distinct_pairs: list, max_ratio: float = 25.0):
    """Balance training data to avoid overwhelming one class.
    
    If matches outnumber non-matches by more than max_ratio, subsample matches.
    """
    if len(distinct_pairs) == 0:
        print("WARNING: No distinct pairs found, model may not train well")
        return match_pairs, distinct_pairs
    
    ratio = len(match_pairs) / len(distinct_pairs)
    
    if ratio > max_ratio:
        target_matches = int(len(distinct_pairs) * max_ratio)
        print(f"Balancing: reducing matches from {len(match_pairs)} to {target_matches}")
        random.seed(42)
        match_pairs = random.sample(match_pairs, target_matches)
    else:
        print(f"No balancing needed (ratio={ratio:.1f})")
    
    return match_pairs, distinct_pairs


def train():
    """Train dedupe model."""
    print("Loading data...")
    left_data, right_data, fields = load_data()
    print(f"  Records: {len(left_data)} left, {len(right_data)} right")
    
    # Remove old settings
    if SETTINGS_FILE.exists():
        os.remove(SETTINGS_FILE)
        print("Removed old settings file")
    
    # Initialize dedupe
    print("\nInitializing dedupe...")
    field_definitions = [dedupe.variables.String("name")]
    linker = dedupe.RecordLink(field_definitions)
    
    # Prepare training - this indexes the records
    print("Preparing training data...")
    linker.prepare_training(left_data, right_data)
    
    # Now load training pairs using the EXACT objects from left_data/right_data
    print("\nLoading training pairs...")
    match_pairs = []
    distinct_pairs = []
    skipped = 0
    
    with open(VALIDATION_DIR / "ground_truth.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left_id = int(row["left_id"])
            right_id = int(row["right_id"])
            is_match = int(row["is_match"])
            
            if left_id not in left_data or right_id not in right_data:
                continue
            
            # Use exact object references from the indexed data
            left_rec = left_data[left_id]
            right_rec = right_data[right_id]
            
            # Skip non-ASCII records (dedupe Unicode bug)
            try:
                left_rec["name"].encode("ascii")
                right_rec["name"].encode("ascii")
            except UnicodeEncodeError:
                skipped += 1
                continue
            
            if is_match == 1:
                match_pairs.append((left_rec, right_rec))
            else:
                distinct_pairs.append((left_rec, right_rec))
    
    print(f"  Matches: {len(match_pairs)}")
    print(f"  Non-matches: {len(distinct_pairs)}")
    print(f"  Skipped (non-ASCII): {skipped}")
    
    # Balance if needed
    match_pairs, distinct_pairs = balance_training_data(match_pairs, distinct_pairs)
    print(f"\nAfter balancing:")
    print(f"  Matches: {len(match_pairs)}")
    print(f"  Non-matches: {len(distinct_pairs)}")
    
    # Mark pairs
    print("\nMarking pairs...")
    training_data = {"match": match_pairs, "distinct": distinct_pairs}
    linker.mark_pairs(training_data)
    
    # Train
    print("Training model...")
    linker.train()
    
    # Save settings
    print(f"Saving model to {SETTINGS_FILE}...")
    with open(SETTINGS_FILE, "wb") as f:
        linker.write_settings(f)
    
    # Quick test
    print("\nTesting model...")
    result = linker.join(left_data, right_data, 0.5)
    print(f"  Found {len(result)} matches at threshold 0.5")
    
    # Check score distribution
    if result:
        scores = [score for (_, _), score in result]
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
    
    print("\nTraining complete!")
    return linker


if __name__ == "__main__":
    train()
