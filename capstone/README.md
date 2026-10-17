# TST Jurisprudence Analysis Pipeline - Capstone Project

## Overview

This capstone project builds an end-to-end data pipeline over the **Livro de
Jurisprudência do TST** (the official compendium of the Brazilian Superior Labor
Court), turning a 579-page PDF into a structured, searchable, quality-controlled
knowledge base.

The pipeline ingests the PDF and segments it **verbete by verbete** (Súmulas,
Orientações Jurisprudenciais and Precedentes Normativos), keeping only the
**current wording** of each verbete. It then generates multilingual embeddings,
stores them in a vector database, organizes them with dimensionality reduction
and clustering, flags low-quality/malformed records, detects anomalies, and
exposes the whole corpus through semantic search and analytics commands.

**Main goals:**

- Convert an unstructured legal PDF into a clean, machine-readable dataset
  (JSONL / Parquet).
- Enable **semantic search** by meaning rather than keywords, across Portuguese
  legal text.
- Surface **structure** in the corpus (themes/clusters) and **quality issues**
  (cancelled, duplicated, truncated or noisy verbetes).
- Provide a reproducible, tested, CI-verified pipeline.

## Domain Statement

- **Domain:** Brazilian labor law (*Direito do Trabalho*), specifically the
  jurisprudence of the **Tribunal Superior do Trabalho (TST)**.
- **Corpus:** the official *Livro de Súmulas, Orientações Jurisprudenciais e
  Precedentes Normativos* ([official source](https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns)).
- **Entity modeled:** a **verbete**, i.e. a single numbered normative statement,
  carrying a type (`sumula`, `oj`, `precedente`), an issuing body (`orgao`), a
  code (`codigo`) and a theme (`tema`).
- **Problem addressed:** the source is a large, Portuguese-language, legally
  dense PDF where verbetes are interleaved with their full historical redactions
  and cross-references, making manual lookup and analysis slow. This project
  converts it into a normalized, searchable, deduplicated dataset.
- **Intended users:** legal researchers, law students, and professionals who need
  to discover related precedents, explore themes, or audit corpus quality.
- **Scope and limitations:**
  - Quality labels are **heuristic** and used for data hygiene, **not** legal
    analysis. Cancellation/revocation is detected from header keywords
    (`cancelad`, `revogad`, etc.) and may not capture every case.
  - The embedding model is general-purpose multilingual; it is not fine-tuned on
    legal language.
  - This project is an educational data-engineering exercise and **does not
    provide legal advice**.

## Architecture

```
Ingestion (PDF → verbetes) → Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
→ ChromaDB (HNSW, cosine) → UMAP + HDBSCAN → Quality Control (LogReg + IsolationForest)
→ JSONL Export + CLI Search/Analytics
```

A detailed component diagram is available as PlantUML source below:

```plantuml
@startuml
title TST Jurisprudence Analysis Pipeline - Architecture
left to right direction
skinparam shadowing false
skinparam componentStyle rectangle
skinparam arrowColor #455A64

actor "Researcher" as user

file "TST PDF" as pdf
database "Parquet" as parquet
database "ChromaDB\n(HNSW · cosine)" as chroma
file "JSONL" as jsonl

rectangle "CLI - cli.main" {
  [ingest] as ingest
  [embed] as embed
  [cluster] as cluster
  [quality] as quality
  [export] as export
  [search] as search
}

pdf --> ingest
ingest --> parquet
parquet --> embed
embed --> chroma
chroma --> cluster
cluster --> chroma
chroma --> quality
quality --> chroma
chroma --> export
export --> jsonl
user --> search
search --> chroma
@enduml
```

Each verbete is segmented from the PDF keeping only the current wording (the book
also reproduces the full history of each verbete), cutting at
`Histórico:`/repeated headers and stopping at the `Índice Remissivo` (Subject
Index).

## Requirements

- Python 3.12+
- 16GB+ RAM (CPU-only by default)
- 10GB+ free disk (model weights + vector store)
- No GPU required (`EMBEDDING_DEVICE=cpu`)

## Configuration

Configuration is centralized in [`src/config.py`](src/config.py) and powered by
`pydantic-settings`, so every value can be overridden through **environment
variables** or a **`.env`** file placed in `capstone/` (not versioned):

```bash
# capstone/.env (optional)
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu
COLLECTION_NAME=tst_verbetes
ANOMALY_CONTAMINATION=0.05
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer used to embed verbetes |
| `EMBEDDING_DIM` | `384` | Embedding dimension |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `TST_PDF_PATH` | `data/raw/livrointernet12pdf.pdf` | Input PDF path |
| `VERBETE_MIN_TEXT` | `20` | Drop verbetes shorter than this |
| `VERBETE_RUIDO` / `VERBETE_CURTO` | `30` / `120` | Thresholds for `ruido`/`curto` labels |
| `COLLECTION_NAME` | `tst_verbetes` | ChromaDB collection name |
| `HNSW_SPACE` | `cosine` | Vector index distance metric |
| `UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST` | `15`, `0.1` | UMAP projection |
| `HDBSCAN_MIN_CLUSTER_SIZE`, `HDBSCAN_MIN_SAMPLES` | `15`, `5` | HDBSCAN clustering |
| `ANOMALY_CONTAMINATION` | `0.05` | IsolationForest contamination |
| `EXPORT_PATH` | `data/final/tst.jsonl` | JSONL export destination |

Project directories (`data/raw`, `data/processed`, `data/final`, `models`,
`chroma_db`) are created automatically on import.

## Setup

From the repository root:

```bash
cd capstone

# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Upgrade packaging tools
pip install --upgrade pip setuptools

# 3. Install the package with dev dependencies
pip install -e ".[dev]"
```

Download the source PDF from the [official page](https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns)
and place it at `data/raw/livrointernet12pdf.pdf` (it is not versioned).

Run the full pipeline:

```bash
capstone pipeline -i data/raw/livrointernet12pdf.pdf
```

## Data

- Official source: [Book of Precedents, OJs and PNs](https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns)
- File: `data/raw/livrointernet12pdf.pdf` (579 pages, not versioned)
- Extracted corpus: ~1058 verbetes (~703 OJs, 235 Súmulas, 120 Precedents)
- A small offline fixture (`tests/fixtures/tst-sample.pdf`, 22 verbetes) is
  committed so that CI can run the pipeline without network access.

`ingest` accepts either a PDF (`-i file.pdf`) or a plain-text file
(`-i file.txt`), detected by the file extension. Plain text is treated as one
verbete per non-empty line, which makes it easy to feed pre-extracted or
manually curated corpora through the same downstream stages.

### Licensing and attribution

- **Code:** released under the MIT License (see [`LICENSE`](LICENSE)).
- **Data:** the source document, the TST *Livro de Súmulas, Orientações
  Jurisprudenciais e Precedentes Normativos*, is an official public act published
  by the Tribunal Superior do Trabalho. Under Brazilian Law 9.610/98, art. 8, IV,
  official acts are **not protected by copyright**. The source PDF is **not
  redistributed** in this repository; it is downloaded from the official TST page
  above. All extracted text is the property of its original public source and is
  used here for research and educational purposes only.

## Usage

### CLI

```bash
# Full pipeline (ingest -> embed -> cluster -> quality -> export)
capstone pipeline -i data/raw/livrointernet12pdf.pdf

# Individual steps
capstone ingest -i data/raw/livrointernet12pdf.pdf -o data/processed/tst.parquet
capstone embed --batch-size 32 --id-prefix verbete
capstone cluster
capstone quality --contamination 0.05
capstone export -o data/final/tst.jsonl

# Search and analytics
capstone search "adicional de insalubridade" --k 10 --tipo oj --orgao SBDI-1
capstone clusters --top-terms 5
capstone anomalies --limit 20
capstone stats
```

`capstone pipeline` accepts `--skip-ingest`, `--skip-embed`, `--skip-cluster`,
`--skip-quality` and `--skip-export` to resume from any stage.

### Command parameters

| Command | Parameter | Meaning | Default |
|---------|-----------|---------|---------|
| `search` | `QUERY` (positional) | Text to search for by meaning | required |
| `search` | `-k`, `--k` | Number of results to return, ranked by similarity score (0 to 1) | `10` |
| `search` | `--tipo` | Keep only this verbete type (`sumula`, `oj`, `precedente`) | all |
| `search` | `--orgao` | Keep only this issuing body (`TST`, `SBDI-1`, `SBDI-2`, `SDC`, `TP-OE`) | all |
| `search` | `--quality` | Keep only this quality label (`completo`, `cancelado`, `curto`, `duplicado`) | all |
| `embed` | `-i`, `--input` | Parquet file produced by `ingest` | `data/processed/tst.parquet` |
| `embed` | `-b`, `--batch-size` | Number of texts embedded per forward pass | `32` |
| `embed` | `--device` | Compute device (`cpu` or `cuda`) | `cpu` |
| `embed` | `--id-prefix` | Prefix for stored record IDs (e.g. `verbete_0`) | `verbete` |
| `cluster` | `--umap-neighbors` | UMAP `n_neighbors` (larger keeps more global structure) | `15` |
| `cluster` | `--umap-min-dist` | UMAP `min_dist` (smaller packs points tighter) | `0.1` |
| `cluster` | `--hdbscan-min-cluster` | Minimum points to form a cluster | `15` |
| `cluster` | `--hdbscan-min-samples` | HDBSCAN density smoothing | `5` |
| `quality` | `--contamination` | Expected outlier ratio for IsolationForest (0.05 = 5%) | `0.05` |
| `clusters` | `--top-terms` | Number of frequent terms shown per cluster | `5` |
| `anomalies` | `-l`, `--limit` | Number of anomalies to display | `20` |
| `anomalies` | `--min-score` | Only show anomalies at or above this score | `0.0` |
| `ingest` | `-i`, `--input` | Source TST PDF | `data/raw/livrointernet12pdf.pdf` |
| `ingest` | `-o`, `--output` | Output Parquet path | `data/processed/tst.parquet` |
| `ingest` | `--min-len` | Discard verbetes shorter than this many characters | `20` |
| `export` | `-o`, `--output` | Output JSONL path | `data/final/tst.jsonl` |
| all | `--collection` | ChromaDB collection name | `tst_verbetes` |

### Makefile shortcuts

```bash
make install    # pip install -e ".[dev]"
make pipeline   # full pipeline
make ingest embed cluster quality export
make search clusters anomalies stats
make dashboard  # launch the Streamlit interface
make docs       # regenerate figures + reports (PDF and PPTX)
make test lint clean rebuild
```

## Dashboard (Streamlit + Plotly)

An interactive interface for exploring the corpus without the CLI:

```bash
capstone-dashboard          # or: make dashboard / streamlit run src/app/dashboard.py
```

It reads `data/final/tst.jsonl` (and the ChromaDB collection, when available) and
provides three tabs:

- **Search** — semantic (or lexical fallback) search with type/quality filters and
  result inspection.
- **Cluster map** — UMAP scatter colored by cluster, plus per-cluster size and top
  terms.
- **Quality & anomalies** — label distribution, text-length histogram, anomaly
  score distribution and a table of the most anomalous verbetes.

## Reports

Reproducible figures and reports are generated from the live artifacts, as both
PDF (document) and PPTX (slides):

```bash
make docs       # -> docs/figures/*.png, docs/report.pdf, docs/qc-report.pdf,
                #    docs/report.pptx, docs/qc-report.pptx
```

- [`docs/report.md`](docs/report.md) — technical report (architecture, results,
  limitations, reproduction steps).
- [`docs/qc-report.md`](docs/qc-report.md) — data-quality-control report (labels,
  classifier evaluation, anomalies, clusters).
- [`docs/figures/`](docs/figures/) — figures and a machine-readable `summary.json`.

PDF rendering uses `pandoc` with the WeasyPrint engine (`--pdf-engine=weasyprint`).
PPTX rendering uses the same `pandoc` sources with `--slide-level=2`.

## Presentation

The demo deck and the spoken script live alongside the reports:

```bash
make slides     # -> docs/slides.pdf and docs/slides.pptx (Marp, via npx)
```

- [`docs/slides.md`](docs/slides.md) — Marp deck (concepts, architecture, results,
  annotated references).
- [`docs/demo-script.md`](docs/demo-script.md) — timed walkthrough, figure-reading
  guide and likely questions with answers.
- `docs/report.pptx` and `docs/qc-report.pptx` — slide versions of the two reports.

Slide rendering uses `@marp-team/marp-cli` through `npx` (needs network the first
time).

## Docker

A self-contained image builds the package and runs the pipeline once on the
offline fixture, so the dashboard starts with data and needs no network access:

```bash
docker compose up --build      # from the repository root -> http://localhost:8501
```

The image is defined in [`Dockerfile`](Dockerfile); the Compose file lives at the
repository root (`docker-compose.yml`).

## JSONL Export

Each line of `data/final/tst.jsonl` is an enriched verbete:

```json
{
  "id": "verbete_0",
  "doc_id": "sumula_6",
  "tipo": "sumula",
  "orgao": "TST",
  "codigo": 6,
  "tema": "Quadro de carreira. Homologação. Equiparação salarial",
  "text": "...",
  "text_len": 353,
  "source": "livrointernet12pdf.pdf",
  "quality_label": "completo",
  "cluster_id": 12,
  "anomaly_score": 0.374641,
  "umap_x": 5.487961,
  "umap_y": 9.684173
}
```

## Heuristic Labels (Quality Control)

| Label | Rule |
|-------|------|
| cancelado (cancelled) | header indicates `cancelad`, `(negativo)`, `revogad`, `cassad`, `superad` |
| duplicado (duplicate) | normalized body repeated across more than one verbete |
| ruido (noise) | text shorter than 30 characters |
| curto (short) | text between 30 and 119 characters |
| completo (complete) | text with 120+ characters |

## Embedding Model

- **paraphrase-multilingual-MiniLM-L12-v2** (118M params, 384-dim, multilingual)

## Clustering

- UMAP: n_neighbors=15, min_dist=0.1, metric=cosine
- HDBSCAN: min_cluster_size=15, min_samples=5, metric=euclidean

## Anomaly

- IsolationForest: n_estimators=200, contamination=0.05

## Project Structure

```
capstone/
├── src/
│   ├── cli/main.py          # Click CLI (ingest, embed, cluster, quality, search, export, pipeline)
│   ├── config.py            # pydantic-settings configuration
│   ├── ingest/tst.py        # PDF/TXT → verbetes (segmentation + heuristic labels)
│   ├── embed/model.py       # SentenceTransformer embeddings
│   ├── store/client.py      # ChromaDB client/collection
│   ├── cluster/pipeline.py  # UMAP + HDBSCAN
│   ├── quality/classifier.py# LogReg quality classifier
│   ├── quality/anomaly.py   # IsolationForest anomaly detection
│   ├── search/engine.py     # Semantic search + analytics
│   ├── export/jsonl.py      # JSONL export
│   ├── app/dashboard.py     # Streamlit + Plotly exploration interface
│   └── app/main.py          # Dashboard launcher (capstone-dashboard)
├── scripts/                 # Report figure generation
├── docs/                    # Technical report, QC report, figures (PDFs)
├── tests/                   # Pytest suite + offline PDF fixture
├── Dockerfile               # Container image (pipeline + dashboard)
└── LICENSE                  # MIT (code) + data licensing note
```

## Code Quality

```bash
black .
mypy .
pytest tests/
```

CI runs `black --check`, `mypy`, the test suite, and a mini offline ETL on every
push to `main`.
