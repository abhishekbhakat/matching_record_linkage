# Matching Algorithm Benchmark

Comparing **5 PPRL implementations** using isolated virtual environments against a 673-row validation dataset.

## Structure

```
MATCHING_TEST/
├── benchmark.py                   # Main runner (isolated venvs)
├── .venvs/                        # Isolated virtual environments
├── clkhash_matcher/               # CLKHash (legacy, archived 2023)
├── polars_bloom_jaccard/          # Polars Bloom (custom lightweight)
├── modular_pprl/                  # Modular PPRL (blooms+recordlinkage)
├── rapidfuzz_matcher/             # RapidFuzz (token_set similarity)
├── splink_matcher/                # Splink (unsupervised probabilistic)
└── validation/                    # Ground truth dataset (673 rows)
```

## Quick Start

```bash
make setup              # Create isolated venvs for all 5 matchers
make benchmark          # Run benchmark on validation dataset
make benchmark-csv      # Run and save results to CSV
make clean              # Remove venvs and cache
```

## Why Isolated Environments?

- **CLKHash** requires `setuptools`, `cryptography`, and other dependencies
- **Polars Bloom** only needs `polars` and `bitarray`
- Avoids version conflicts between packages
- Each implementation runs in its own Python process

## Implementations

### CLKHash (from Prefect Flows)

Uses the official `clkhash` library with:
- Schema-based field specifications
- Key derivation for cryptographic hashing
- `stream_bloom_filters` for encoding
- Bitarray-based Jaccard similarity

### Polars Bloom Jaccard

Lightweight implementation:
- Direct MD5 hashing of n-grams
- Simple Bloom filter encoding
- Same Jaccard similarity calculation
- ~10x faster due to reduced overhead

## Benchmark Results

**Dataset:** 673 rows, 452,929 comparisons  
**Ground truth:** 467 true matches, 206 non-matches

| Matcher | Time | Matches | Precision | Recall | F1 |
|---------|------|---------|-----------|--------|-----|
| **CLKHash** | 0.15s | 552 | 0.71 | **0.84** | **0.77** |
| **Polars Bloom** | **0.10s** | 511 | 0.72 | 0.79 | 0.75 |
| Modular PPRL | 23.16s | 420 | 0.64 | 0.57 | 0.60 |
| RapidFuzz | 0.44s | 665 | 0.58 | 0.82 | 0.68 |
| Splink | 0.59s | 8 | 0.00 | 0.00 | 0.00 |

**Winners:**
- **Speed:** Polars Bloom (0.10s)
- **F1 Score:** CLKHash (0.77)

**Metrics:**
- **Precision** = correct matches / predicted matches (higher = fewer false positives)
- **Recall** = correct matches / 467 ground truth (higher = fewer missed matches)
- **F1** = harmonic mean of both (higher = better overall)

*Example: CLKHash returns 552 matches, ~400 are correct*
- Precision = 400/552 = 0.72 (72% of predictions are correct)
- Recall = 400/481 = 0.83 (83% of true matches found)
- High recall + lower precision = finds most matches but includes some false positives

## Manual Setup

```bash
# Create CLKHash venv
uv venv .venvs/clkhash_matcher
uv pip install -p .venvs/clkhash_matcher -r clkhash_matcher/requirements.txt

# Create Polars venv
uv venv .venvs/polars_bloom_jaccard
uv pip install -p .venvs/polars_bloom_jaccard -r polars_bloom_jaccard/requirements.txt

# Run individual benchmarks
.venvs/clkhash_matcher/bin/python clkhash_matcher/run_benchmark.py data.json 0.5
.venvs/polars_bloom_jaccard/bin/python polars_bloom_jaccard/run_benchmark.py data.json 0.5
```

## Thresholds

Each matcher has a `config.yaml` with tuned thresholds:

| Matcher | Threshold |
|---------|----------|
| CLKHash | 0.47 |
| Polars Bloom | 0.46 |
| Modular PPRL | 0.57 |
| Splink | 0.50 |
| RapidFuzz | 0.70 |

### Splink Note

Splink is a probabilistic record linkage tool that requires EM training to estimate m/u parameters. Without proper training data, it produces very few matches. The current benchmark shows Splink's limitations when used without training - it's designed for large-scale deduplication where patterns can be learned from data.

## Important: Always Use `uv`

All Python commands must use `uv`:

```bash
uv run python <script.py>
```

Never use bare `python` or `python3` commands.