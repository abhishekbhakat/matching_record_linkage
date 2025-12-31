"""Modular PPRL: blooms encoding + Polars.

Minimal dependencies:
1. blooms: Privacy-preserving Bloom filter encoding
2. polars: Fast DataFrame operations
"""

import polars as pl
from blooms import blooms as BloomFilter


BLOOM_SIZE = 1024
NGRAM_SIZE = 2


def _generate_ngrams(text: str, n: int) -> list[str]:
    text = text.lower().strip()
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def encode_to_bloom(value: str, ngram_size: int = NGRAM_SIZE) -> list[int]:
    """Encode a string value into Bloom filter bit list."""
    bf = BloomFilter(BLOOM_SIZE)
    ngrams = _generate_ngrams(str(value), ngram_size)
    for ngram in ngrams:
        bf @= ngram.encode()
    return [int(b) for b in bf]


def bloom_jaccard(b1: list[int], b2: list[int]) -> float:
    """Compute Jaccard similarity on Bloom filter bits."""
    inter = sum(a & b for a, b in zip(b1, b2))
    union = sum(a | b for a, b in zip(b1, b2))
    return inter / union if union > 0 else 0.0


def modular_pprl_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
    ngram_size: int = NGRAM_SIZE,
) -> pl.DataFrame:
    """Modular PPRL matching using blooms + Polars."""
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    left_blooms_dict = {}
    right_blooms_dict = {}

    for field in fields:
        if field in df_left.columns:
            left_blooms_dict[field] = [
                encode_to_bloom(str(v) if v is not None else "", ngram_size)
                for v in df_left[field].to_list()
            ]
        if field in df_right.columns:
            right_blooms_dict[field] = [
                encode_to_bloom(str(v) if v is not None else "", ngram_size)
                for v in df_right[field].to_list()
            ]

    matches = []
    n_left, n_right = len(df_left), len(df_right)

    for i in range(n_left):
        for j in range(n_right):
            weighted_sum = 0.0
            for field in fields:
                if field in left_blooms_dict and field in right_blooms_dict:
                    sim = bloom_jaccard(left_blooms_dict[field][i], right_blooms_dict[field][j])
                    weighted_sum += weights.get(field, 1.0) * sim

            score = weighted_sum / total_weight
            if score >= threshold:
                matches.append({"left_idx": i, "right_idx": j, "score": round(score, 4)})

    if not matches:
        return pl.DataFrame({
            "left_idx": pl.Series([], dtype=pl.Int64),
            "right_idx": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(matches)
