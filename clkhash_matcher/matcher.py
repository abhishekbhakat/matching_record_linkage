"""CLKHash matcher - copied from prefect flows blind_match.py."""

import polars as pl
from bitarray import bitarray

from encoder import encode_per_field


def bloom_jaccard(bf1: bytes, bf2: bytes) -> float:
    """Compute Jaccard similarity between two Bloom filters."""
    ba1 = bitarray()
    ba1.frombytes(bf1)
    ba2 = bitarray()
    ba2.frombytes(bf2)

    inter = (ba1 & ba2).count()
    union = (ba1 | ba2).count()
    return inter / union if union > 0 else 0.0


def pprl_fuzzy_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
    secret: str = "default_secret",
) -> pl.DataFrame:
    """Privacy-preserving fuzzy match using per-field Bloom filters.

    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        fields: List of column names to match on
        weights: Dict of {field: weight} for weighted scoring
        threshold: Minimum similarity score [0..1]
        secret: Secret key for hashing

    Returns:
        DataFrame with columns: left_idx, right_idx, score
    """
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    enc_left = encode_per_field(df_left, fields, secret=secret)
    enc_right = encode_per_field(df_right, fields, secret=secret)

    matches = []
    for i in range(len(df_left)):
        for j in range(len(df_right)):
            weighted_sum = 0.0
            for field in fields:
                bf1 = enc_left[field][i]
                bf2 = enc_right[field][j]
                sim = bloom_jaccard(bf1, bf2)
                weighted_sum += weights.get(field, 1.0) * sim

            score = weighted_sum / total_weight
            if score >= threshold:
                matches.append({
                    "left_idx": i,
                    "right_idx": j,
                    "score": round(score, 4),
                })

    if not matches:
        return pl.DataFrame({
            "left_idx": pl.Series([], dtype=pl.Int64),
            "right_idx": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(matches)
