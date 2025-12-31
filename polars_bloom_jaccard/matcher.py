"""Polars-based Bloom filter Jaccard matching - CLKHash-compatible encoder.

Recreates CLKHash's encoding without the unmaintained library dependency.
Uses same algorithm: n-grams + double hashing + BitsPerTokenStrategy.
"""

import hashlib
import hmac

import polars as pl
from bitarray import bitarray

BLOOM_SIZE = 1024
BITS_PER_TOKEN = 5
NGRAM_SIZE = 2
SECRET = b"default_secret"


def _derive_keys(secret: bytes, num_keys: int = 2) -> list[bytes]:
    """Derive hash keys from secret (matches CLKHash key_derivation)."""
    keys = []
    for i in range(num_keys):
        key = hmac.new(secret, f"key{i}".encode(), hashlib.sha256).digest()
        keys.append(key)
    return keys


def _double_hash(token: str, keys: list[bytes], bloom_size: int) -> list[int]:
    """Double hashing to get bit positions (matches CLKHash doubleHash)."""
    token_bytes = token.encode()
    h1 = int.from_bytes(hmac.digest(keys[0], token_bytes, "sha256")[:8], "big")
    h2 = int.from_bytes(hmac.digest(keys[1], token_bytes, "sha256")[:8], "big")

    return [(h1 + i * h2) % bloom_size for i in range(BITS_PER_TOKEN)]


def _generate_ngrams(text: str, n: int) -> list[str]:
    """Generate n-grams from text."""
    text = text.lower().strip()
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def string_to_bloom(
    s: str,
    n: int = NGRAM_SIZE,
    keys: list[bytes] | None = None,
) -> bitarray:
    """Encode string into Bloom filter (CLKHash-compatible)."""
    bf = bitarray(BLOOM_SIZE)
    bf.setall(0)

    if keys is None:
        keys = _derive_keys(SECRET)

    ngrams = _generate_ngrams(s, n)
    for ngram in ngrams:
        positions = _double_hash(ngram, keys, BLOOM_SIZE)
        for pos in positions:
            bf[pos] = 1

    return bf


def bloom_jaccard(bf1: bitarray, bf2: bitarray) -> float:
    """Compute Jaccard similarity on Bloom filters (bitwise AND/OR)."""
    inter = (bf1 & bf2).count()
    union = (bf1 | bf2).count()
    return inter / union if union > 0 else 0.0


def encode_bloom_filters(
    df: pl.DataFrame,
    fields: list[str],
    ngram_size: int = NGRAM_SIZE,
    secret: bytes = SECRET,
) -> dict[str, list[bitarray]]:
    """Encode each field into per-row Bloom filters."""
    encodings: dict[str, list[bitarray]] = {}
    keys = _derive_keys(secret)

    for field in fields:
        if field not in df.columns:
            encodings[field] = [bitarray(BLOOM_SIZE) for _ in range(len(df))]
            for bf in encodings[field]:
                bf.setall(0)
            continue

        col_encodings = []
        for row in df.iter_rows(named=True):
            val = str(row.get(field, "") or "")
            bf = string_to_bloom(val, ngram_size, keys)
            col_encodings.append(bf)

        encodings[field] = col_encodings

    return encodings


def polars_bloom_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
    ngram_size: int = NGRAM_SIZE,
    secret: bytes = SECRET,
) -> pl.DataFrame:
    """Polars-based Bloom filter Jaccard matching.

    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        fields: List of column names to match on
        weights: Dict of {field: weight} for weighted scoring
        threshold: Minimum similarity score [0..1]
        ngram_size: Size of n-grams for Bloom filter encoding

    Returns:
        DataFrame with columns: left_idx, right_idx, score
    """
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    enc_left = encode_bloom_filters(df_left, fields, ngram_size, secret)
    enc_right = encode_bloom_filters(df_right, fields, ngram_size, secret)

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
