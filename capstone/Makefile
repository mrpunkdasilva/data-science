.PHONY: install ingest embed cluster quality search stats anomalies clusters pipeline export jsonl test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python -m cli.main ingest -i data/raw/livrointernet12pdf.pdf -o data/processed/tst.parquet

embed:
	python -m cli.main embed --batch-size 32 --id-prefix verbete

cluster:
	python -m cli.main cluster

quality:
	python -m cli.main quality

search:
	@read -p "Query: " q; python -m cli.main search "$$q" --k 10

clusters:
	python -m cli.main clusters

anomalies:
	python -m cli.main anomalies --limit 20

stats:
	python -m cli.main stats

pipeline:
	python -m cli.main pipeline -i data/raw/livrointernet12pdf.pdf

jsonl:
	python -m cli.main export

test:
	pytest tests/ -v --tb=short

lint:
	black . && mypy .

clean:
	rm -rf data/processed/* chroma_db/* models/*

rebuild: clean pipeline
	@echo "Full rebuild complete!"