# Capstone CSDS-352 - Navegador de documentos científicos

Navegador interativo para um corpus de papers do arXiv. A ideia é simples de
descrever: o sistema lê os papers, transforma cada um em um vetor numérico, guarda
esses vetores num banco vetorial local e usa essa representação para buscar,
agrupar e avaliar a qualidade dos documentos.

Tudo aqui foi escrito à mão, sem depender de um framework de vector database
(Chroma, FAISS, Weaviate) ou de um serviço de embeddings na nuvem.

## O que o sistema faz

| Requisito | Onde está |
|-----------|-----------|
| Ingestão de texto, Markdown e PDF | `src/ingest/` |
| Coleta de papers do arXiv | `src/ingest/arxiv_fetcher.py` |
| Embeddings locais (MiniLM e TF-IDF/SVD) | `src/embeddings/` |
| Banco vetorial com metadados | `src/index/vector_store.py` |
| Busca semântica e mais-parecidos | `src/index/search.py` |
| Clusterização e projeção 2D | `src/cluster/` |
| Qualidade de documentos e anomalias | `src/quality/` |
| API em Python e CLI | `src/api/` |
| Interface web | `src/ui/streamlit_app.py` |

## Como rodar

### 1. Instalar

TF-IDF é imediato e não precisa de GPU nem de download de modelo. Para os embeddings neurais, da interface
web ou do UMAP, use os extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[embeddings,ui,umap]'
```

### 2. Baixar o corpus

```bash
capstone ingest --category cs.LG --max-results 400
```

Isso grava um `.txt` por paper (título + resumo, que é o que entra no
embedding) e um `manifest.jsonl` com os metadados estruturados: autores,
categorias, data de publicação e links. Os 400 papers já estão versionados em
`data/corpus/`, então este passo é opcional.

### 3. Construir o índice

```bash
capstone build --backend tfidf
capstone build --backend minilm
```

O `tfidf` não depende de download de modelo nenhum e roda em segundos. O
`minilm` usa `all-MiniLM-L6-v2` (384 dimensões) e produz resultados de busca
melhores, com custo de download de ~90 MB na primeira vez.

### 4. Analisar

```bash
capstone cluster --n-clusters 8 --projection umap
```

Produz os clusters, as coordenadas 2D, o score de qualidade e a lista de
anomalias, tudo gravado em `data/index/analysis.json` para a interface ler.

### 5. Explorar

```bash
capstone search "como redes neurais aprendem representações"
capstone explore
capstone serve
```

## API em Python

A mesma funcionalidade está disponível como uma classe, para usar em notebook
ou script:

```python
from api.service import CapstoneAPI

api = CapstoneAPI()                    # usa capstone/data por padrão
api.build(backend="tfidf")             # devolve um dicionário de resumo
api.analyze(backend="tfidf")           # clusters, projeção 2D, qualidade

for hit in api.search("self-supervised vision", k=5):
    print(f"{hit.score:.3f}  {hit.title}")

# mais-parecidos com um documento que já está no índice
outros = api.similar(hit.doc_id, k=3)
```

`search` e `similar` usam automaticamente o mesmo backend com que o índice foi
construído, então não é preciso passar `backend=` a cada chamada.

## Decisões de projeto

### Banco vetorial escrito à mão

`VectorStore` guarda os vetores normalizados em um único `embeddings.npy` e os
metadados em JSON. Como os vetores são normalizados (norma 1), a similaridade de
cosseno vira um único produto matricial: `V @ q`. O `argpartition` acha os `k`
melhores em tempo linear e só então ordenamos esse subconjunto.

Isso evita uma dependência pesada e deixa o formato do índice legível em git.

### TF-IDF/SVD e MiniLM, não só um

O fallback TF-IDF + SVD foi uma escolha prática: ele roda offline, sem GPU e em
segundos, o que importa para o professor conseguir executar o projeto. O TF-IDF
captura palavras-chave, então funciona bem para papers. O MiniLM captura
sinônimos e parafrases, o que ajuda em buscas conceituais.

### PCA guarda pouco da variância, e isso está explícito

Com 256 dimensões do TF-IDF, as duas primeiras componentes do PCA guardam cerca
de **2% da variância** (`variance_explained` no `analysis.json`). Ou seja: a
visualização é uma simplificação agressiva, e o sistema diz isso na interface em
vez de esconder. Para mapa com separação visual melhor, use `--projection umap`,
que preserva vizinhanças em vez de variância.

### Qualidade e anomalias são regras transparentes

O score de qualidade usa regras que dá para explicar em voz alta: tamanho do
documento, presença de seções esperadas em um paper, legibilidade das frases e
presença de resumo. Nada de modelo treinado em cima de rótulo que ninguém
auditou. O classificador supervisionado (`RandomForest`) existe e está testado,
mas o pipeline usa a heurística como padrão.

Anomalias usam Isolation Forest: um paper cujo vetor está longe de todos os
outros é raro, e "raro" neste contexto significa off-topic, quebrado ou
duplicado.

## Testes

```bash
pytest capstone/tests -q
```

41 testes cobrem ingestão (incluindo o merge do manifest), o vector store, a
busca, os componentes de ML e a interface do Streamlit. Os testes de UI usam o
`AppTest` do próprio Streamlit e são pulados quando o extra de UI não está
instalado.

## Limitações conhecidas

- O corpus do repositório usa a query `cat:cs.LG`, que traz papers que têm
  `cs.LG` em qualquer uma de suas categorias. Os papers vêm de várias áreas
  (cs.AI, stat.ML, cs.CL, cs.CV, cs.RO), o que é bom para a clusterização, mas
  não é um recorte temático limpo.
- Só o texto (título e resumo) é ingerido. Os PDFs não são baixados.
- A projeção 2D do UMAP é recalculada a cada execução, sem cache.
- `analysis.json` é reescrito a cada `capstone cluster`.
