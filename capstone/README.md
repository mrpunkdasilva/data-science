# Amazon Reviews Analysis Pipeline - Capstone Project

Sistema de análise de reviews Amazon usando embeddings, clusterização e controle de qualidade.

## Arquitetura

```
Ingestão (HF Datasets) → Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
→ ChromaDB (HNSW, cosine) → UMAP + HDBSCAN → Quality Control (LogReg + IsolationForest)
→ CLI Search/Analytics
```

## Comandos CLI

```bash
# Pipeline completo
capstone pipeline --limit 15000

# Passos individuais
capstone ingest --limit 15000
capstone embed --batch-size 32
capstone cluster
capstone quality --contamination 0.05

# Busca e análise
capstone search "broken screen" --k 10 --category Electronics
capstone clusters --top-terms 5
capstone anomalies --limit 20
capstone stats
```

## Requisitos

- Python 3.12+
- 16GB+ RAM (CPU only)
- 10GB+ disco livre

## Instalação

```bash
pip install -e ".[dev]"
```

## Qualidade de Código

```bash
black .
mypy .
pytest tests/
```

## Modelo de Embedding

- **paraphrase-multilingual-MiniLM-L12-v2** (118M params, 384-dim, multilíngue)
- Roda em CPU (~2k docs/s no i5-1235U)

## Dados

- Fonte: HuggingFace `McAuley-Lab/Amazon-Reviews-2023`
- Categorias: Electronics, Home_and_Kitchen
- Filtros: verified_purchase, text_len 50-500

## Labels Heurísticos (Quality Control)

| Label | Regra |
|-------|-------|
| helpful | helpful_vote ≥ 10 AND rating ≥ 4 AND len > 100 |
| funny | helpful_vote ≥ 5 AND keywords (funny, lol, 😂, haha) |
| weird | rating ≤ 2 AND helpful_vote ≥ 5 AND keywords (weird, strange) |
| fake_suspect | rating = 5 AND helpful_vote = 0 AND len < 80 AND marketing keywords |
| normal | resto |

## Clusterização

- UMAP: n_neighbors=15, min_dist=0.1, metric=cosine
- HDBSCAN: min_cluster_size=15, min_samples=5, metric=euclidean

## Anomalia

- IsolationForest: n_estimators=200, contamination=0.05
