.PHONY: install ingest embed cluster quality search stats anomalies clusters pipeline test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python -m cli.main ingest --limit 15000

embed:
	python -m cli.main embed --batch-size 32

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
	python -m cli.main pipeline --limit 15000

test:
	pytest tests/ -v --tb=short

lint:
	black . && mypy .

clean:
	rm -rf data/processed/* chroma_db/* models/*

rebuild: clean pipeline
	@echo "Full rebuild complete!"