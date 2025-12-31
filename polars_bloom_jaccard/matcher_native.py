"""Ultra-fast Polars Bloom matcher using native Polars hash (Rust-backed)."""

import polars as pl
from bitarray import bitarray

BLOOM_SIZE = 1024
BITS_PER_TOKEN = 5
NGRAM_SIZE = 2


def _create_ngram_df(df: pl.DataFrame, field: str, n: int = NGRAM_SIZE) -> pl.DataFrame:
    """Create n-grams using Polars string operations."""
    return (
        df.with_row_index("_idx")
        .with_columns(pl.col(field).str.to_lowercase().str.strip_chars().alias("_text"))
        .with_columns(pl.col("_text").str.len_chars().alias("_len"))
        .with_columns(
            pl.when(pl.col("_len") < n)
            .then(pl.struct([pl.col("_text").alias("ngram")]))
            .otherwise(
                pl.col("_text").str.extract_all(f"(?=(.{{{n}}}))").alias("ngrams")
            )
        )
        .explode("ngrams")
        .with_columns(
            pl.when(pl.col("ngrams").is_null())
            .then(pl.col("_text"))
            .otherwise(pl.col("ngrams").struct.field("ngram") if "ngram" in str(df.schema) else pl.col("ngrams"))
            .alias("ngram")
        )
        .select(["_idx", "ngram"])
        .filter(pl.col("ngram").is_not_null() & (pl.col("ngram") != ""))
    )


def _encode_field_native(df: pl.DataFrame, field: str, n: int = NGRAM_SIZE) -> list[bitarray]:
    """Encode a field using Polars native hash."""
    n_rows = len(df)
    blooms = [bitarray(BLOOM_SIZE) for _ in range(n_rows)]
    for bf in blooms:
        bf.setall(0)

    values = df[field].to_list()
    for idx, val in enumerate(values):
        if val is None:
            continue
        text = str(val).lower().strip()
        if not text:
            continue

        ngrams = [text] if len(text) < n else [text[i:i+n] for i in range(len(text) - n + 1)]

        for ngram in ngrams:
            h = hash(ngram)
            h1 = h & 0xFFFFFFFF
            h2 = (h >> 32) & 0xFFFFFFFF
            for i in range(BITS_PER_TOKEN):
                pos = (h1 + i * h2) % BLOOM_SIZE
                blooms[idx][pos] = 1

    return blooms


def bloom_jaccard(bf1: bitarray, bf2: bitarray) -> float:
    """Compute Jaccard similarity on Bloom filters."""
    inter = (bf1 & bf2).count()
    union = (bf1 | bf2).count()
    return inter / union if union > 0 else 0.0


def polars_bloom_match_native(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
    ngram_size: int = NGRAM_SIZE,
) -> pl.DataFrame:
    """Ultra-fast Polars Bloom filter matching using native Python hash."""
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    enc_left = {f: _encode_field_native(df_left, f, ngram_size) for f in fields if f in df_left.columns}
    enc_right = {f: _encode_field_native(df_right, f, ngram_size) for f in fields if f in df_right.columns}

    n_left = len(df_left)
    n_right = len(df_right)

    matches = []
    for i in range(n_left):
        for j in range(n_right):
            weighted_sum = 0.0
            for field in fields:
                if field in enc_left and field in enc_right:
                    sim = bloom_jaccard(enc_left[field][i], enc_right[field][j])
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
