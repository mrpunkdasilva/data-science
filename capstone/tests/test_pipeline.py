"""Tests for capstone pipeline."""

import pytest
import numpy as np
from config import settings


def test_config_loads():
    """Test that settings load correctly."""
    assert (
        settings.EMBEDDING_MODEL
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert settings.EMBEDDING_DIM == 384
    assert settings.COLLECTION_NAME == "tst_verbetes"
    assert settings.TST_PDF_PATH.name == "livrointernet12pdf.pdf"
    assert settings.VERBETE_MIN_TEXT == 20
    assert settings.VERBETE_CURTO == 120
    assert settings.VERBETE_RUIDO == 30


def test_embedding_dim():
    """Test embedding dimension matches config."""
    from embed.model import get_model

    model = get_model()
    emb = model.encode(["test"], normalize_embeddings=True)
    assert emb.shape == (1, settings.EMBEDDING_DIM)


def test_chroma_client():
    """Test ChromaDB client creation."""
    from store.client import get_chroma_client

    client = get_chroma_client()
    assert client is not None


def test_segment_verbetes():
    """Test verbete segmentation from cleaned PDF lines, incl. historical repeats."""
    from ingest.tst import _segment_verbetes

    lines = [
        "Nº 386 Cobrança de haveres. Ex-sócio. Data de saída da sociedade",
        "Texto da súmula 386 sobre a data de saída da sociedade.",
        "SÚMULA 386",
        "Nº 386 Cobrança de haveres. Ex-sócio.",
        "OJ-SDI1-92 - 1988. Adicional de insalubridade",
        "Texto da OJ SDI1 92 de teste.",
        "PN-1 (negativo) - Cancelamento de precedente. Tema repetitivo",
        "Texto do PN 1 de teste.",
        "Seção 2",
    ]
    verbetes = _segment_verbetes(lines)
    tipos = {v["tipo"]: v for v in verbetes}
    assert set(tipos) == {"sumula", "oj", "precedente"}
    assert tipos["sumula"]["codigo"] == 386
    assert "Cobrança" in tipos["sumula"]["tema"]
    assert tipos["oj"]["codigo"] == 92
    assert tipos["oj"]["orgao"] == "SBDI-1"
    assert tipos["precedente"]["codigo"] == 1
    assert "(negativo)" in tipos["precedente"]["tema"]


def test_opening_heading_split():
    """Test that the 'Nº <num> <tema>' heading yields both fields and merging works."""
    from ingest.tst import HEADING_SUMULA

    m = HEADING_SUMULA.match(
        "Nº 6 Quadro de carreira. Homologação. Equiparação salarial"
    )
    assert m is not None
    assert int(m.group(1)) == 6
    assert "Homologação" in m.group(2)

    m2 = HEADING_SUMULA.match("Súmula 386 Cobrança de haveres")
    assert m2 is not None
    assert int(m2.group(1)) == 386
    assert "Cobrança" in m2.group(2)

    assert HEADING_SUMULA.match("Nº 00492-67.2011.5.17.0002") is None


def test_label_verbetes():
    """Test heuristic quality labeling for verbetes."""
    from ingest.tst import _label_verbetes

    verbetes = [
        {
            "tipo": "sumula",
            "orgao": "TST",
            "codigo": 100,
            "tema": "Tema completo",
            "body": [
                "Conteudo longo e substancial do verbete um, com varias",
                "frases de texto juridico que compoem um verbete completo",
                "e de qualidade para analise jurisprudencial posterior.",
            ],
        },
        {
            "tipo": "sumula",
            "orgao": "TST",
            "codigo": 200,
            "tema": "Tema curto",
            "body": ["Súmula cancelada."],
        },
        {
            "tipo": "oj",
            "orgao": "SBDI-1",
            "codigo": 300,
            "tema": "Ruido",
            "body": ["x"],
        },
        {
            "tipo": "oj",
            "orgao": "SBDI-1",
            "codigo": 400,
            "tema": "Duplicado A",
            "body": ["Mesmo corpo duplicado de texto completo e longo"],
        },
        {
            "tipo": "oj",
            "orgao": "SBDI-1",
            "codigo": 401,
            "tema": "Duplicado B",
            "body": ["Mesmo corpo duplicado de texto completo e longo"],
        },
    ]
    _label_verbetes(verbetes)
    by_code = {v["codigo"]: v for v in verbetes}
    assert by_code[100]["quality_label"] == "completo"
    assert by_code[200]["quality_label"] == "cancelado"
    assert by_code[300]["quality_label"] == "ruido"
    assert by_code[400]["quality_label"] == "duplicado"
    assert by_code[400]["duplicate_count"] == 2
    assert by_code[401]["duplicate_count"] == 2


def test_isolation_forest():
    """Test anomaly detection basic functionality."""
    from sklearn.ensemble import IsolationForest

    X = np.random.randn(100, 384)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    # Use score_samples which returns negative log-likelihood (higher = more normal)
    # Negate to get anomaly score (higher = more anomalous)
    scores = -clf.score_samples(X)
    # Normalize to 0-1 range
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    assert len(scores) == 100
    assert scores.min() >= 0
    assert scores.max() <= 1


def test_export_record():
    """Test that a ChromaDB entry becomes a proper JSONL record."""
    from export.jsonl import _build_record

    record = _build_record(
        "verbete_0",
        {
            "doc_id": "sumula_6",
            "tipo": "sumula",
            "orgao": "TST",
            "codigo": 6,
            "tema": "Quadro de carreira. Homologação",
            "text": "A homologação por escrito é válida.",
            "text_len": 39,
            "source": "livrointernet12pdf.pdf",
            "quality_label": "completo",
            "cluster_id": 2,
            "anomaly_score": 0.123456789,
            "umap_x": 1.5,
            "umap_y": -2.5,
        },
        "A homologação por escrito é válida.",
    )
    assert record["id"] == "verbete_0"
    assert record["doc_id"] == "sumula_6"
    assert record["tipo"] == "sumula"
    assert record["orgao"] == "TST"
    assert record["quality_label"] == "completo"
    assert record["cluster_id"] == 2
    assert record["anomaly_score"] == 0.123457
    assert record["umap_x"] == 1.5
    assert record["umap_y"] == -2.5


def test_export_jsonl_roundtrip(tmp_path):
    """Export a temp collection to JSONL and verify line count + valid JSON."""
    import json as jsonlib
    from typing import Any
    from export.jsonl import export_jsonl
    from store.client import get_chroma_client

    client: Any = get_chroma_client()
    coll_name = "test_tst_export"
    if [c for c in client.list_collections() if c.name == coll_name]:
        client.delete_collection(coll_name)
    collection = client.create_collection(name=coll_name)

    collection.add(
        ids=["verbete_0", "verbete_1"],
        documents=["Texto súmula um de teste.", "Texto OJ dois de teste."],
        metadatas=[
            {
                "doc_id": "sumula_1",
                "tipo": "sumula",
                "orgao": "TST",
                "codigo": 1,
                "tema": "Tema um",
                "text": "Texto súmula um de teste.",
                "text_len": 24,
                "source": "tst-sample.pdf",
                "quality_label": "completo",
                "cluster_id": 0,
                "anomaly_score": 0.1,
            },
            {
                "doc_id": "oj_sbdi1_2",
                "tipo": "oj",
                "orgao": "SBDI-1",
                "codigo": 2,
                "tema": "Tema dois",
                "text": "Texto OJ dois de teste.",
                "text_len": 22,
                "source": "tst-sample.pdf",
                "quality_label": "curto",
                "cluster_id": 0,
                "anomaly_score": 0.2,
            },
        ],
    )

    out = tmp_path / "tst.jsonl"
    n = export_jsonl(collection_name=coll_name, output_path=str(out))
    client.delete_collection(coll_name)

    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n
    first = jsonlib.loads(lines[0])
    assert first["tipo"] == "sumula"
    assert "text" in first
    assert "quality_label" in first
    assert "anomaly_score" in first


def test_logistic_regression():
    """Test classifier basic functionality."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["good product", "bad product", "great item", "terrible item"]
    y = ["positive", "negative", "positive", "negative"]
    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(X)
    clf = LogisticRegression(random_state=42)
    clf.fit(X_vec, y)
    pred = clf.predict(vec.transform(["awesome product"]))
    assert pred[0] == "positive"


def test_load_verbetes_from_plain_text(tmp_path):
    """Plain-text ingestion parses the same verbete types as the PDF reader."""
    from ingest.tst import load_verbetes

    content = "\n".join(
        [
            "SÚMULA 1 Futuro do adicional de insalubridade",
            "Texto da súmula um sobre insalubridade em grau médio e suas"
            " consequências legais para o trabalhador.",
            "OJ-SDI1-92 - 1988. Adicional de insalubridade",
            "Texto da orientação jurisprudencial noventa e dois com corpo"
            " suficientemente longo para não ser descartado pelo limite.",
            "PN-1 (negativo) - Cancelamento de precedente. Tema repetitivo",
            "Texto do precedente normativo um com corpo suficientemente longo"
            " para permanecer no resultado da ingestão de texto.",
        ]
    )
    source = tmp_path / "verbetes.txt"
    source.write_text(content, encoding="utf-8")

    df = load_verbetes(source_path=source, min_text_len=20)

    assert set(df["tipo"]) == {"sumula", "oj", "precedente"}
    assert df.loc[df["tipo"] == "oj", "orgao"].iloc[0] == "SBDI-1"
    assert (df["source"] == "verbetes.txt").all()


def test_load_verbetes_rejects_unknown_format(tmp_path):
    """An unsupported extension raises a clear error."""
    from ingest.tst import load_verbetes

    source = tmp_path / "verbetes.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_verbetes(source_path=source)


def test_search_reviews_filters():
    """Semantic search supports single and combined metadata filters.

    ChromaDB requires an explicit $and when more than one condition is given,
    so combining tipo + orgao + quality_label must not raise.
    """
    from typing import Any
    from search.engine import search_reviews
    from store.client import get_chroma_client

    client: Any = get_chroma_client()
    coll_name = "test_search_filters"
    if [c for c in client.list_collections() if c.name == coll_name]:
        client.delete_collection(coll_name)
    collection = client.create_collection(name=coll_name)

    base = {
        "orgao": "TST",
        "tema": "Adicional de insalubridade",
        "text_len": 40,
        "source": "tst-sample.pdf",
        "cluster_id": 0,
        "anomaly_score": 0.1,
    }
    collection.add(
        ids=["v0", "v1", "v2"],
        documents=[
            "Súmula sobre adicional de insalubridade em grau médio.",
            "OJ sobre adicional de insalubridade para o trabalhador.",
            "Precedente normativo sobre adicional de insalubridade.",
        ],
        metadatas=[
            {
                **base,
                "doc_id": "sumula_1",
                "tipo": "sumula",
                "quality_label": "completo",
            },
            {**base, "doc_id": "oj_1", "tipo": "oj", "quality_label": "completo"},
            {
                **base,
                "doc_id": "precedente_1",
                "tipo": "precedente",
                "quality_label": "curto",
            },
        ],
    )

    single = search_reviews(
        "adicional de insalubridade", k=5, collection_name=coll_name, tipo="oj"
    )
    assert [r["tipo"] for r in single] == ["oj"]

    combined = search_reviews(
        "adicional de insalubridade",
        k=5,
        collection_name=coll_name,
        tipo="oj",
        orgao="TST",
        quality_label="completo",
    )
    client.delete_collection(coll_name)

    assert len(combined) == 1
    assert combined[0]["tipo"] == "oj"
    assert combined[0]["orgao"] == "TST"
    assert combined[0]["quality_label"] == "completo"


def test_top_terms_filters_noise():
    """Top terms drop stopwords, punctuation and standalone numbers/dates."""
    from lib.text import top_terms

    texts = [
        "para a homologação de sessão res. 20.08.1998) (cancelado",
        "homologação de sessão com sessão",
    ]
    result = top_terms(texts, 5)
    assert "para" not in result
    assert "com" not in result
    assert "res" not in result
    assert all(not any(ch.isdigit() for ch in w) for w in result)
    assert result[0] in {"homologação", "sessão"}


def test_get_anomalies_includes_codigo():
    """Anomaly records carry the 'codigo' field the CLI prints."""
    from typing import Any
    from search.engine import get_anomalies
    from store.client import get_chroma_client

    client: Any = get_chroma_client()
    name = "test_anomalies_codigo"
    if [c for c in client.list_collections() if c.name == name]:
        client.delete_collection(name)
    collection = client.create_collection(name=name)
    collection.add(
        ids=["v0"],
        embeddings=[[0.0] * 384],
        documents=["Texto de teste suficientemente longo."],
        metadatas=[
            {
                "doc_id": "sumula_1",
                "tipo": "sumula",
                "orgao": "TST",
                "codigo": 1,
                "tema": "Tema",
                "quality_label": "completo",
                "cluster_id": 0,
                "anomaly_score": 0.9,
            }
        ],
    )
    results = get_anomalies(collection_name=name, limit=5)
    client.delete_collection(name)

    assert results
    assert results[0]["codigo"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
