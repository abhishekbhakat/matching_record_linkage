"""Ultra-fast Polars Bloom matcher with type-aware ngram encoding.

Matches CLKHash behavior:
- Strings: n=2 (bigrams)
- Dates: n=3 (trigrams)
- Integers: n=3 (trigrams on string representation)

Reads bloom filter parameters from config.yaml.
"""

from pathlib import Path

import polars as pl
import yaml
from bitarray import bitarray

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        _config = yaml.safe_load(f)
    BLOOM_SIZE = _config.get("bloom_filter", {}).get("size", 1024)
    BITS_PER_TOKEN = _config.get("bloom_filter", {}).get("bits_per_token", 5)
else:
    BLOOM_SIZE = 1024
    BITS_PER_TOKEN = 5

# Type-aware ngram sizes (matching CLKHash)
NGRAM_STRING = 2  # For strings
NGRAM_DATE = 3    # For dates
NGRAM_INTEGER = 3 # For integers


def _detect_column_type(df: pl.DataFrame, field: str) -> str:
    """Detect if column is string, date, or integer."""
    if field not in df.columns:
        return "string"
    
    dtype = df[field].dtype
    
    if dtype in (pl.Date, pl.Datetime):
        return "date"
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "integer"
    
    # Check if string column contains date-like values
    if dtype == pl.Utf8 or dtype == pl.String:
        sample = df[field].drop_nulls().head(10).to_list()
        if sample:
            # Check for date patterns like "YYYY-MM-DD" or "MM/DD/YYYY"
            import re
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$")
            date_matches = sum(1 for s in sample if date_pattern.match(str(s)))
            if date_matches >= len(sample) * 0.8:
                return "date"
    
    return "string"


def _get_ngram_size(col_type: str) -> int:
    """Get ngram size based on column type (matching CLKHash)."""
    if col_type == "date":
        return NGRAM_DATE
    elif col_type == "integer":
        return NGRAM_INTEGER
    else:
        return NGRAM_STRING


def _encode_field_native(
    df: pl.DataFrame, 
    field: str, 
    col_type: str | None = None
) -> list[bitarray]:
    """Encode a field using type-aware ngram sizes."""
    n_rows = len(df)
    blooms = [bitarray(BLOOM_SIZE) for _ in range(n_rows)]
    for bf in blooms:
        bf.setall(0)

    if field not in df.columns:
        return blooms

    # Detect column type if not provided
    if col_type is None:
        col_type = _detect_column_type(df, field)
    
    n = _get_ngram_size(col_type)

    values = df[field].to_list()
    for idx, val in enumerate(values):
        if val is None:
            continue
        
        # Format value based on type
        if col_type == "date":
            if hasattr(val, "strftime"):
                text = val.strftime("%Y-%m-%d")
            else:
                text = str(val)
        else:
            text = str(val).lower().strip()
        
        if not text:
            continue

        # Generate ngrams
        ngrams = [text] if len(text) < n else [text[i:i+n] for i in range(len(text) - n + 1)]

        # Hash ngrams into bloom filter
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
) -> pl.DataFrame:
    """Type-aware Polars Bloom filter matching.
    
    Uses different ngram sizes per column type:
    - Strings: n=2 (bigrams)
    - Dates: n=3 (trigrams)
    - Integers: n=3 (trigrams)
    
    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        fields: List of field names to compare
        weights: Dict mapping field name to weight
        threshold: Minimum similarity score
    """
    total_weight = sum(weights.get(f, 1.0) for f in fields)
    if total_weight == 0:
        total_weight = float(len(fields))

    # Detect column types and encode
    enc_left = {}
    enc_right = {}
    for field in fields:
        if field in df_left.columns:
            col_type = _detect_column_type(df_left, field)
            enc_left[field] = _encode_field_native(df_left, field, col_type)
        if field in df_right.columns:
            col_type = _detect_column_type(df_right, field)
            enc_right[field] = _encode_field_native(df_right, field, col_type)

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


# Alias for benchmark compatibility
polars_bloom_match = polars_bloom_match_native
