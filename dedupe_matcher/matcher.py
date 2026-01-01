"""Dedupe-based record linkage using active learning with pre-labeled training data."""

import csv
import os
from pathlib import Path

import dedupe
import dedupe.variables
import polars as pl

MATCHER_DIR = Path(__file__).parent
TRAINING_FILE = MATCHER_DIR / "training_pairs.csv"
SETTINGS_FILE = MATCHER_DIR / "learned_settings"
TRAINING_CACHE = MATCHER_DIR / "training_cache.json"

_linker = None
_is_trained = False


def _load_training_pairs(
    left_data: dict, right_data: dict, fields: list[str]
) -> dict:
    """Load pre-labeled training pairs from CSV.
    
    Args:
        left_data: Dict of {index: {field: value}} from prepare_training
        right_data: Dict of {index: {field: value}} from prepare_training
        fields: List of field names
    """
    match_pairs = []
    distinct_pairs = []

    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left_id = int(row["left_id"])
            right_id = int(row["right_id"])
            is_match = int(row["is_match"])

            if left_id not in left_data or right_id not in right_data:
                continue

            left_record = left_data[left_id]
            right_record = right_data[right_id]

            if is_match == 1:
                match_pairs.append((left_record, right_record))
            else:
                distinct_pairs.append((left_record, right_record))

    return {"match": match_pairs, "distinct": distinct_pairs}


def _get_linker(fields: list[str], df_left: pl.DataFrame, df_right: pl.DataFrame):
    """Get or create the dedupe linker."""
    global _linker, _is_trained

    if _linker is not None and _is_trained:
        return _linker

    field_definitions = [dedupe.variables.String(f) for f in fields]

    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "rb") as f:
            _linker = dedupe.StaticRecordLink(f)
        _is_trained = True
        return _linker

    _linker = dedupe.RecordLink(field_definitions)

    left_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(df_left.to_dicts())
    }
    right_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(df_right.to_dicts())
    }

    _linker.prepare_training(left_data, right_data)

    training_pairs = _load_training_pairs(left_data, right_data, fields)
    _linker.mark_pairs(training_pairs)

    _linker.train()

    with open(SETTINGS_FILE, "wb") as f:
        _linker.write_settings(f)

    _is_trained = True
    return _linker


def dedupe_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Record linkage using dedupe library with pre-labeled training.

    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        fields: List of column names to match on
        weights: Dict of {field: weight} (not used by dedupe, included for interface compatibility)
        threshold: Minimum confidence score [0..1]

    Returns:
        DataFrame with columns: left_idx, right_idx, score
    """
    linker = _get_linker(fields, df_left, df_right)

    left_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(df_left.to_dicts())
    }
    right_data = {
        i: {f: str(row.get(f, "") or "") for f in fields}
        for i, row in enumerate(df_right.to_dicts())
    }

    linked_records = linker.join(left_data, right_data, threshold)

    matches = []
    for (left_id, right_id), score in linked_records:
        matches.append(
            {
                "left_idx": left_id,
                "right_idx": right_id,
                "score": round(score, 4),
            }
        )

    if not matches:
        return pl.DataFrame(
            {
                "left_idx": pl.Series([], dtype=pl.Int64),
                "right_idx": pl.Series([], dtype=pl.Int64),
                "score": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(matches)


def reset_model():
    """Reset the trained model to force retraining."""
    global _linker, _is_trained
    _linker = None
    _is_trained = False
    if SETTINGS_FILE.exists():
        os.remove(SETTINGS_FILE)
    if TRAINING_CACHE.exists():
        os.remove(TRAINING_CACHE)
