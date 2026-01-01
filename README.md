# Matching Algorithm Benchmark

Comparing **6 matching implementations** using isolated virtual environments against a 678-row validation dataset.

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
├── modernbert_matcher/            # ModernBERT-Embed (semantic, Nomic AI)
├── dedupe_matcher/                # Dedupe (active learning, supervised)
└── validation/                    # Ground truth dataset (658 rows)
```

## Quick Start

```bash
make setup              # Create isolated venvs for all matchers
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

**Dataset:** 678 rows, 459,684 comparisons  
**Ground truth:** 657 matches, 21 non-matches

| Matcher | Threshold | Precision | Recall | F1 |
|---------|-----------|-----------|--------|-----|
| **ModernBERT-Embed** | 0.78 | 0.79 | **0.81** | **0.80** |
| CLKHash | 0.41 | **0.79** | 0.77 | 0.78 |
| RapidFuzz | 0.70 | 0.72 | 0.76 | 0.75 |
| Polars Bloom | 0.42 | 0.77 | 0.71 | 0.74 |
| Modular PPRL | 0.54 | 0.65 | 0.51 | 0.57 |
| Splink | 0.50 | — | — | — |

**Winners:**
- **F1 Score:** ModernBERT (0.80)
- **Recall:** ModernBERT (0.81)
- **Precision:** ModernBERT/CLKHash (0.79)

### Production Viability

With the updated ground truth (657 matches, 21 non-matches), **ModernBERT achieves ~79% precision** — reasonable for many use cases but may need human review for critical applications.

**Non-match examples in ground truth:**
- **Similar names:** "Michael Johnson" ≠ "Mitchell Johnson"
- **Spelling variants:** "Sean Murphy" ≠ "Shawn Murphy"
- **Different people:** "Emily Williams" ≠ "Emma Williams"

**Recommendations for production:**
1. Use ModernBERT or CLKHash as primary matcher (best F1)
2. Add human review for borderline cases
3. Consider ensemble approach for higher precision

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

Each matcher has a `config.yaml` with tuned thresholds (optimized via golden section search):

| Matcher | Threshold |
|---------|----------|
| ModernBERT-Embed | 0.78 |
| CLKHash | 0.41 |
| Polars Bloom | 0.42 |
| RapidFuzz | 0.70 |
| Modular PPRL | 0.54 |
| Splink | 0.50 |

### ModernBERT-Embed

Semantic embedding model from Nomic AI (Dec 2024). Uses cosine similarity on 768-dim embeddings. Can match abbreviations like `MIT ↔ Massachusetts Institute of Technology` that string-based matchers miss, but ~60x slower than CLKHash.

### Splink Note

Splink is a probabilistic record linkage tool that requires EM training to estimate m/u parameters. Without proper training data, it produces very few matches.

## Important: Always Use `uv`

All Python commands must use `uv`:

```bash
uv run python <script.py>
```

Never use bare `python` or `python3` commands.