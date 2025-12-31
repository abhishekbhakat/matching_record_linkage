"""RapidFuzz matcher using token_set_ratio for fuzzy string matching."""

import polars as pl
from rapidfuzz import fuzz


def rapidfuzz_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Record linkage using RapidFuzz token_set_ratio."""
    left_data = df_left.to_dicts()
    right_data = df_right.to_dicts()

    matches = []
    threshold_pct = threshold * 100

    for i, left_row in enumerate(left_data):
        for j, right_row in enumerate(right_data):
            total_score = 0.0
            total_weight = 0.0

            for field in fields:
                left_val = str(left_row.get(field, "") or "")
                right_val = str(right_row.get(field, "") or "")

                score = fuzz.token_set_ratio(left_val, right_val)
                weight = weights.get(field, 1.0)
                total_score += score * weight
                total_weight += weight

            final_score = total_score / total_weight if total_weight > 0 else 0

            if final_score >= threshold_pct:
                matches.append({
                    "left_idx": i,
                    "right_idx": j,
                    "score": round(final_score / 100, 4),
                })

    if not matches:
        return pl.DataFrame({
            "left_idx": pl.Series([], dtype=pl.Int64),
            "right_idx": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(matches)
