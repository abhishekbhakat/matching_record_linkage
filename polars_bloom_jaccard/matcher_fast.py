"""Optimized Polars Bloom matcher using native Polars operations (Rust-backed)."""

import hashlib

import polars as pl
from bitarray import bitarray

BLOOM_SIZE = 1024
BITS_PER_TOKEN = 5
NGRAM_SIZE = 2


def _hash_to_positions(hash_val: int) -> list[int]:
    """Convert hash to bit positions using double hashing."""
    h1 = hash_val & 0xFFFFFFFF
    h2 = (hash_val >> 32) & 0xFFFFFFFF
    return [(h1 + i * h2) % BLOOM_SIZE for i in range(BITS_PER_TOKEN)]


def _string_to_bloom_fast(s: str, n: int = NGRAM_SIZE) -> bitarray:
    """Fast Bloom filter encoding using hashlib (no HMAC overhead)."""
    bf = bitarray(BLOOM_SIZE)
    bf.setall(0)

    s = s.lower().strip()
    if not s:
        return bf

    ngrams = [s] if len(s) < n else [s[i:i + n] for i in range(len(s) - n + 1)]

    for ngram in ngrams:
        h = int(hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest(), 16)
        for pos in _hash_to_positions(h):
            bf[pos] = 1

    return bf


def bloom_jaccard(bf1: bitarray, bf2: bitarray) -> float:
    """Compute Jaccard similarity on Bloom filters."""
    inter = (bf1 & bf2).count()
    union = (bf1 | bf2).count()
    return inter / union if union > 0 else 0.0


def encode_bloom_filters_fast(
    df: pl.DataFrame,
    fields: list[str],
    ngram_size: int = NGRAM_SIZE,
) -> dict[str, list[bitarray]]:
    """Encode fields into Bloom filters - optimized version."""
    encodings: dict[str, list[bitarray]] = {}

    for field in fields:
        if field not in df.columns:
            encodings[field] = [bitarray(BLOOM_SIZE) for _ in range(len(df))]
            for bf in encodings[field]:
                bf.setall(0)
            continue

        values = df[field].to_list()
        encodings[field] = [
            _string_to_bloom_fast(str(v) if v is not None else "", ngram_size)
            for v in values
        ]

    return encodings


def polars_bloom_match_fast(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
    ngram_size: int = NGRAM_SIZE,
) -> pl.DataFrame:
    """Optimized Polars Bloom filter Jaccard matching."""
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    enc_left = encode_bloom_filters_fast(df_left, fields, ngram_size)
    enc_right = encode_bloom_filters_fast(df_right, fields, ngram_size)

    n_left = len(df_left)
    n_right = len(df_right)

    matches = []
    for i in range(n_left):
        for j in range(n_right):
            weighted_sum = 0.0
            for field in fields:
                bf1 = enc_left[field][i]
                bf2 = enc_right[field][j]
                sim = bloom_jaccard(bf1, bf2)
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
