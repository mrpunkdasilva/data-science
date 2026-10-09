# TST Jurisprudence Analysis Pipeline - Capstone Project

Pipeline for analyzing the **TST Jurisprudence Book** (Precedents, Jurisprudential
Guidelines and Normative Precedents) using multilingual embeddings, clustering,
semantic search and quality control.

## Architecture

```
Ingestion (PDF → verbetes) → Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
→ ChromaDB (HNSW, cosine) → UMAP + HDBSCAN → Quality Control (LogReg + IsolationForest)
→ JSONL Export + CLI Search/Analytics
```

Each verbete is segmented from the PDF keeping only the current wording (the book
also reproduces the full history of each verbete), cutting at
`Histórico:`/repeated headers and stopping at the `Índice Remissivo` (Subject
Index).

## Requirements

- Python 3.12+
- 16GB+ RAM (CPU only)
- 10GB+ free disk

## Installation

```bash
pip install -e ".[dev]"
```

## Data

- Official source: [Book of Precedents, OJs and PNs](https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns)
- File: `data/raw/livrointernet12pdf.pdf` (579 pages, not versioned)
- Extracted corpus: ~1058 verbetes (~703 OJs, 235 Precedents, 120 Normative Precedents)

## CLI Commands

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

## Code Quality

```bash
black .
mypy .
pytest tests/
```
