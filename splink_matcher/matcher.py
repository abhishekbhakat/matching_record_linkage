"""Splink matcher using DuckDB backend."""

import pandas as pd
import polars as pl
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, DuckDBAPI, block_on


def splink_match(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    fields: list[str],
    weights: dict[str, float],
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Record linkage using Splink with Jaro-Winkler similarity."""
    pd_left = df_left.select(fields).to_pandas().copy()
    pd_right = df_right.select(fields).to_pandas().copy()

    pd_left["unique_id"] = range(len(pd_left))
    pd_right["unique_id"] = range(len(pd_right))

    comparisons = [cl.LevenshteinAtThresholds(f, [1, 2, 3]).configure(term_frequency_adjustments=False) for f in fields]

    settings = SettingsCreator(
        link_type="link_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=[],
    )

    db_api = DuckDBAPI()
    linker = Linker(
        [pd_left, pd_right],
        settings,
        db_api,
        input_table_aliases=["df_left", "df_right"],
    )

    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

    linker.training.estimate_probability_two_random_records_match(
        [block_on(f) for f in fields], recall=0.8
    )

    predictions = linker.inference.predict(threshold_match_probability=threshold)
    results_df = predictions.as_pandas_dataframe()

    matches = []
    for _, row in results_df.iterrows():
        src_l = str(row.get("source_dataset_l", ""))
        src_r = str(row.get("source_dataset_r", ""))

        if "df_left" in src_l and "df_right" in src_r:
            matches.append({
                "left_idx": int(row["unique_id_l"]),
                "right_idx": int(row["unique_id_r"]),
                "score": round(float(row["match_probability"]), 4),
            })
        elif "df_right" in src_l and "df_left" in src_r:
            matches.append({
                "left_idx": int(row["unique_id_r"]),
                "right_idx": int(row["unique_id_l"]),
                "score": round(float(row["match_probability"]), 4),
            })

    if not matches:
        return pl.DataFrame({
            "left_idx": pl.Series([], dtype=pl.Int64),
            "right_idx": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(matches)
