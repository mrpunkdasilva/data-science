# TST Jurisprudence Analysis Pipeline - Capstone Project

Pipeline de análise do **Livro de Jurisprudência do TST** (Súmulas, Orientações
Jurisprudenciais e Precedentes Normativos) usando embeddings multilíngues,
clusterização, busca semântica e controle de qualidade.

## Arquitetura

```
Ingestão (PDF → verbetes) → Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
→ ChromaDB (HNSW, cosine) → UMAP + HDBSCAN → Quality Control (LogReg + IsolationForest)
→ Export JSONL + CLI Search/Analytics
```

Cada verbete é segmentado do PDF mantendo apenas a redação vigente (o livro
também reproduz o histórico completo de cada verbete), com corte em
`Histórico:`/repetições de cabeçalho e parada no `Índice Remissivo`.

## Requisitos

- Python 3.12+
- 16GB+ RAM (CPU only)
- 10GB+ disco livre

## Instalação

```bash
pip install -e ".[dev]"
```

## Dados

- Fonte oficial: [Livro de Súmulas, OJs e PNs](https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns)
- Arquivo: `data/raw/livrointernet12pdf.pdf` (579 páginas, não versionado)
- Corpus extraído: ~1058 verbetes (~703 OJs, 235 Súmulas, 120 Precedentes Normativos)

## Comandos CLI

```bash
# Pipeline completo (ingest -> embed -> cluster -> quality -> export)
capstone pipeline -i data/raw/livrointernet12pdf.pdf

# Passos individuais
capstone ingest -i data/raw/livrointernet12pdf.pdf -o data/processed/tst.parquet
capstone embed --batch-size 32 --id-prefix verbete
capstone cluster
capstone quality --contamination 0.05
capstone export -o data/final/tst.jsonl

# Busca e análise
capstone search "adicional de insalubridade" --k 10 --tipo oj --orgao SBDI-1
capstone clusters --top-terms 5
capstone anomalies --limit 20
capstone stats
```

## Export JSONL

Cada linha de `data/final/tst.jsonl` é um verbete enriquecido:

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

## Labels Heurísticos (Quality Control)

| Label | Regra |
|-------|-------|
| cancelado | cabeçalho indica `cancelad`, `(negativo)`, `revogad`, `cassad`, `superad` |
| duplicado | corpo normalizado repetido em mais de um verbete |
| ruido | texto com menos de 30 caracteres |
| curto | texto entre 30 e 119 caracteres |
| completo | texto com 120+ caracteres |

## Modelo de Embedding

- **paraphrase-multilingual-MiniLM-L12-v2** (118M params, 384-dim, multilíngue)

## Clusterização

- UMAP: n_neighbors=15, min_dist=0.1, metric=cosine
- HDBSCAN: min_cluster_size=15, min_samples=5, metric=euclidean

## Anomalia

- IsolationForest: n_estimators=200, contamination=0.05

## Qualidade de Código

```bash
black .
mypy .
pytest tests/
```
