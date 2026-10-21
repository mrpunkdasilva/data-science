"""Semantic search and analytics engine."""

# mypy: ignore-errors

import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from config import settings
from store.client import get_chroma_client, get_collection
from embed.model import get_model
from lib.text import top_terms as compute_top_terms
from chromadb.api.models.Collection import Collection
from chromadb.types import Where


def search_reviews(
    query: str,
    k: int = 10,
    collection_name: str = settings.COLLECTION_NAME,
    tipo: str | None = None,
    orgao: str | None = None,
    quality_label: str | None = None,
) -> list[dict]:
    """
    Semantic search for similar verbetes.

    Args:
        query: Search query text
        k: Number of results to return
        collection_name: ChromaDB collection name
        tipo: Filter by tipo (sumula/oj/precedente)
        orgao: Filter by orgão emisor (TST/SBDI-1/SBDI-2/SDC/TP-OE)
        quality_label: Filter by quality label

    Returns:
        List of verbete dicts with metadata and similarity score
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    model = get_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    # Build where filter (Chroma needs $and for multiple conditions)
    conditions: list[dict] = []
    if tipo:
        conditions.append({"tipo": tipo})
    if orgao:
        conditions.append({"orgao": orgao})
    if quality_label:
        conditions.append({"quality_label": quality_label})

    where: Where | None = None
    if len(conditions) == 1:
        where = conditions[0]  # type: ignore[assignment]
    elif len(conditions) > 1:
        where = {"$and": conditions}  # type: ignore[assignment]

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, settings.MAX_SEARCH_K),
        where=where,
        include=["documents", "metadatas", "distances"],  # type: ignore[arg-type]
    )

    formatted: list[dict] = []
    ids = results["ids"]
    metadatas = results["metadatas"]
    documents = results["documents"]
    distances = results["distances"]

    if ids and metadatas and documents and distances:
        for i in range(len(ids[0])):
            meta = metadatas[0][i]
            formatted.append(
                {
                    "id": ids[0][i],
                    "doc_id": meta.get("doc_id", ids[0][i]),
                    "tipo": meta.get("tipo", ""),
                    "orgao": meta.get("orgao", ""),
                    "codigo": meta.get("codigo", ""),
                    "tema": meta.get("tema", ""),
                    "text": documents[0][i],
                    "quality_label": meta.get("quality_label", "completo"),
                    "cluster_id": meta.get("cluster_id", -1),
                    "anomaly_score": meta.get("anomaly_score", 0.0),
                    "score": 1.0 - distances[0][i],  # cosine similarity
                }
            )

    return formatted


def list_clusters(
    collection_name: str = settings.COLLECTION_NAME,
    top_terms: int = 5,
) -> list[dict]:
    """
    List all clusters with statistics.

    Returns:
        List of cluster info dicts
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    count = collection.count()
    result = collection.get(include=["metadatas", "documents"], limit=count)  # type: ignore[arg-type]

    metadatas = result["metadatas"] or []
    documents = result["documents"] or []

    clusters: dict[int, dict[str, list]] = {}
    for meta, doc in zip(metadatas, documents):
        cid = int(meta.get("cluster_id", -1))
        if cid not in clusters:
            clusters[cid] = {
                "texts": [],
                "qualities": [],
                "tipos": [],
            }
        # Use the body text from metadata so cluster descriptors match the
        # dashboard and the report (the Chroma document is tema + " " + text).
        clusters[cid]["texts"].append(meta.get("text", doc))
        clusters[cid]["qualities"].append(meta.get("quality_label", "completo"))
        clusters[cid]["tipos"].append(meta.get("tipo", ""))

    cluster_info: list[dict] = []
    for cid, data in sorted(clusters.items()):
        if cid == -1:
            continue  # skip noise
        size = len(data["tipos"])
        dominant_quality = max(set(data["qualities"]), key=data["qualities"].count)
        tipo_counts = Counter(data["tipos"]).most_common(3)

        top_words = compute_top_terms(data["texts"], top_terms)

        cluster_info.append(
            {
                "cluster_id": cid,
                "size": size,
                "dominant_quality": dominant_quality,
                "top_terms": top_words,
                "tipos": dict(tipo_counts),
            }
        )

    return cluster_info


def get_anomalies(
    collection_name: str = settings.COLLECTION_NAME,
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Get top anomalous verbetes.

    Returns:
        List of verbete dicts sorted by anomaly_score descending
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    count = collection.count()
    result = collection.get(include=["metadatas", "documents"], limit=count)  # type: ignore[arg-type]

    metadatas = result["metadatas"] or []
    documents = result["documents"] or []
    ids = result["ids"] or []

    anomalies: list[dict] = []
    for i, (meta, doc, uid) in enumerate(zip(metadatas, documents, ids)):
        score = float(meta.get("anomaly_score", 0.0))
        if score >= min_score:
            anomalies.append(
                {
                    "id": uid,
                    "doc_id": meta.get("doc_id", uid),
                    "tema": meta.get("tema", ""),
                    "text": doc,
                    "tipo": meta.get("tipo", ""),
                    "orgao": meta.get("orgao", ""),
                    "codigo": meta.get("codigo", ""),
                    "quality_label": meta.get("quality_label", "completo"),
                    "cluster_id": meta.get("cluster_id", -1),
                    "anomaly_score": score,
                }
            )

    anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return anomalies[:limit]


def get_stats(collection_name: str = settings.COLLECTION_NAME) -> dict:
    """Get collection statistics."""
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    count = collection.count()
    result = collection.get(include=["metadatas"], limit=count)  # type: ignore[arg-type]

    metadatas = result["metadatas"] or []

    tipos: dict[str, int] = {}
    orgaos: dict[str, int] = {}
    quality_dist: dict[str, int] = {}
    cluster_ids: set[int] = set()
    anomaly_scores: list[float] = []

    for m in metadatas:
        t = str(m.get("tipo", "unknown"))
        tipos[t] = tipos.get(t, 0) + 1

        o = str(m.get("orgao", "unknown"))
        orgaos[o] = orgaos.get(o, 0) + 1

        qlabel = str(m.get("quality_label", "completo"))
        quality_dist[qlabel] = quality_dist.get(qlabel, 0) + 1

        cid = int(m.get("cluster_id", -1))
        cluster_ids.add(cid)

        anomaly_scores.append(float(m.get("anomaly_score", 0.0)))

    n_clusters = len([c for c in cluster_ids if c != -1])
    noise_count = sum(1 for m in metadatas if int(m.get("cluster_id", -1)) == -1)
    anomaly_count = sum(1 for s in anomaly_scores if s > 0.7)

    return {
        "collection": collection_name,
        "total_documents": count,
        "tipos": tipos,
        "orgaos": orgaos,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "quality_distribution": quality_dist,
        "anomaly_count": anomaly_count,
        "anomaly_pct": (anomaly_count / count * 100) if count > 0 else 0,
    }
