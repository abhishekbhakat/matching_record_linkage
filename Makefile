.PHONY: benchmark benchmark-csv setup clean help

help:
	@echo "Matching Algorithm Benchmark (5 matchers)"
	@echo ""
	@echo "Matchers (thresholds from config.yaml):"
	@echo "  1. CLKHash (legacy)      - threshold=0.47"
	@echo "  2. Polars Bloom          - threshold=0.45"
	@echo "  3. Modular PPRL          - threshold=0.55"
	@echo "  4. Splink                - threshold=0.50"
	@echo "  5. RapidFuzz             - threshold=0.70"
	@echo ""
	@echo "Dataset: validation/source_left.csv, source_right.csv (673 rows)"
	@echo ""
	@echo "Usage:"
	@echo "  make setup              Create isolated venvs for all 5 matchers"
	@echo "  make benchmark          Run benchmark on validation dataset"
	@echo "  make benchmark-csv      Run and save results to CSV"
	@echo "  make clean              Remove venvs and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make setup && make benchmark"

benchmark:
	./benchmark.py

benchmark-csv:
	./benchmark.py -o results.csv

setup:
	@echo "Creating CLKHash venv..."
	uv venv .venvs/clkhash_matcher
	uv pip install -p .venvs/clkhash_matcher -r clkhash_matcher/requirements.txt
	@echo "Creating Polars Bloom venv..."
	uv venv .venvs/polars_bloom_jaccard
	uv pip install -p .venvs/polars_bloom_jaccard -r polars_bloom_jaccard/requirements.txt
	@echo "Creating Modular PPRL venv..."
	uv venv .venvs/modular_pprl
	uv pip install -p .venvs/modular_pprl -r modular_pprl/requirements.txt
	@echo "Creating Splink venv..."
	uv venv .venvs/splink_matcher
	uv pip install -p .venvs/splink_matcher -r splink_matcher/requirements.txt
	@echo "Creating RapidFuzz venv..."
	uv venv .venvs/rapidfuzz_matcher
	uv pip install -p .venvs/rapidfuzz_matcher -r rapidfuzz_matcher/requirements.txt
	@echo "Done. All 5 matchers ready."

clean:
	rm -rf .venvs
	rm -f results.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
