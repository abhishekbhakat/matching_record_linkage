"""CLKHash encoder - copied from prefect flows blind_match.py."""

import datetime

import polars as pl
from bitarray import bitarray
from clkhash import field_formats
from clkhash.bloomfilter import stream_bloom_filters
from clkhash.comparators import get_comparator
from clkhash.key_derivation import generate_key_lists
from clkhash.schema import Schema


def make_spec(col: str, df: pl.DataFrame) -> field_formats.FieldSpec:
    """Choose the right FieldSpec based on the column dtype."""
    dtype = df[col].dtype

    is_date = dtype in (pl.Date, pl.Datetime)

    if is_date:
        return field_formats.DateSpec(
            identifier=col,
            hashing_properties=field_formats.FieldHashingProperties(
                comparator=get_comparator({"type": "ngram", "n": 3}),
                strategy=field_formats.BitsPerTokenStrategy(5),
                encoding="utf-8",
                hash_type="doubleHash",
                prevent_singularity=False,
            ),
            format="%Y-%m-%d",
        )

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return field_formats.IntegerSpec(
            identifier=col,
            hashing_properties=field_formats.FieldHashingProperties(
                comparator=get_comparator({"type": "numeric", "thresholdDistance": 1, "resolution": 2}),
                strategy=field_formats.BitsPerFeatureStrategy(10),
                encoding="utf-8",
                hash_type="doubleHash",
                prevent_singularity=False,
            ),
        )

    return field_formats.StringSpec(
        identifier=col,
        hashing_properties=field_formats.FieldHashingProperties(
            comparator=get_comparator({"type": "ngram", "n": 2}),
            strategy=field_formats.BitsPerTokenStrategy(5),
            encoding="utf-8",
            hash_type="doubleHash",
            prevent_singularity=False,
        ),
    )


def encode_per_field(
    df: pl.DataFrame,
    fields: list[str],
    bit_length: int = 1024,
    secret: str | bytes = "default_secret",
) -> dict[str, list[bytes]]:
    """Bloom-filter-encode each field separately using CLKHash."""
    encodings: dict[str, list[bytes]] = {}

    for field in fields:
        if field not in df.columns:
            zero = bitarray(bit_length)
            zero.setall(False)
            encodings[field] = [zero.tobytes() for _ in range(len(df))]
            continue

        spec = make_spec(field, df)
        schema = Schema(fields=[spec], l=bit_length, xor_folds=0)
        key_lists = generate_key_lists(secret, num_identifier=1)

        records = []
        for row in df.iter_rows(named=True):
            val = row.get(field)
            if val is None:
                text = ""
            elif isinstance(val, (datetime.date, datetime.datetime)):
                text = val.strftime("%Y-%m-%d")
            else:
                text = str(val)
            records.append([text])

        valid_record_tuples = [(i, rec) for i, rec in enumerate(records) if rec[0] != ""]
        blank_indices = {i for i, rec in enumerate(records) if rec[0] == ""}
        valid_records = [rec for _, rec in valid_record_tuples]

        if valid_records:
            bf_iter = stream_bloom_filters(valid_records, key_lists, schema)
            bloom_map = {i: bf.tobytes() for (i, _), (bf, _, _) in zip(valid_record_tuples, bf_iter)}
        else:
            bloom_map = {}

        enc_list: list[bytes] = []
        for i in range(len(records)):
            if i in blank_indices:
                zero = bitarray(bit_length)
                zero.setall(False)
                enc_list.append(zero.tobytes())
            else:
                enc_list.append(bloom_map[i])

        encodings[field] = enc_list

    return encodings
