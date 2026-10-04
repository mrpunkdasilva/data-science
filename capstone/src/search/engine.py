"""Semantic search and analytics engine."""

# mypy: ignore-errors

import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from config import settings
from store.client import get_chroma_client, get_collection
from embed.model import get_model
from chromadb.api.models.Collection import Collection
from chromadb.types import Where


def search_reviews(
    query: str,
    k: int = 10,
    collection_name: str = "amazon_reviews",
    category: str | None = None,
    rating_min: int | None = None,
    quality_label: str | None = None,
) -> list[dict]:
    """
    Semantic search for similar reviews.

    Args:
        query: Search query text
        k: Number of results to return
        collection_name: ChromaDB collection name
        category: Filter by category (Electronics, Home_Kitchen)
        rating_min: Minimum rating (1-5)
        quality_label: Filter by quality label

    Returns:
        List of review dicts with metadata and similarity score
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    model = get_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    # Build where filter
    where: Where = {}
    if category:
        where["category"] = category
    if rating_min:
        where["rating"] = {"$gte": rating_min}
    if quality_label:
        where["quality_label"] = quality_label

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, settings.MAX_SEARCH_K),
        where=where if where else None,
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
                    "review_id": ids[0][i],
                    "title": meta.get("title", ""),
                    "text": documents[0][i],
                    "category": meta.get("category", ""),
                    "rating": meta.get("rating", 0),
                    "helpful_vote": meta.get("helpful_vote", 0),
                    "quality_label": meta.get("quality_label", "normal"),
                    "cluster_id": meta.get("cluster_id", -1),
                    "anomaly_score": meta.get("anomaly_score", 0.0),
                    "score": 1.0 - distances[0][i],  # cosine similarity
                }
            )

    return formatted


def list_clusters(
    collection_name: str = "amazon_reviews",
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

    # Group by cluster_id
    clusters: dict[int, dict[str, list]] = {}
    for meta, doc in zip(metadatas, documents):
        cid = int(meta.get("cluster_id", -1))
        if cid not in clusters:
            clusters[cid] = {"reviews": [], "ratings": [], "helpful": [], "qualities": []}
        clusters[cid]["reviews"].append(doc)
        clusters[cid]["ratings"].append(meta.get("rating", 0))
        clusters[cid]["helpful"].append(meta.get("helpful_vote", 0))
        clusters[cid]["qualities"].append(meta.get("quality_label", "normal"))

    # Compute stats
    cluster_info: list[dict] = []
    for cid, data in sorted(clusters.items()):
        if cid == -1:
            continue  # skip noise
        size = len(data["ratings"])
        avg_rating = np.mean(data["ratings"])
        avg_helpful = np.mean(data["helpful"])
        dominant_quality = max(set(data["qualities"]), key=data["qualities"].count)

        # Simple top terms (most frequent words in cluster)
        all_text = " ".join(data["reviews"]).lower()
        words = [w for w in all_text.split() if len(w) > 3]
        top_words = [w for w, _ in Counter(words).most_common(top_terms)]

        cluster_info.append(
            {
                "cluster_id": cid,
                "size": size,
                "avg_rating": round(avg_rating, 2),
                "avg_helpful": round(avg_helpful, 1),
                "dominant_quality": dominant_quality,
                "top_terms": top_words,
            }
        )

    return cluster_info


def get_anomalies(
    collection_name: str = "amazon_reviews",
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Get top anomalous reviews.

    Returns:
        List of review dicts sorted by anomaly_score descending
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
                    "review_id": uid,
                    "title": meta.get("title", ""),
                    "text": doc,
                    "category": meta.get("category", ""),
                    "rating": meta.get("rating", 0),
                    "helpful_vote": meta.get("helpful_vote", 0),
                    "quality_label": meta.get("quality_label", "normal"),
                    "cluster_id": meta.get("cluster_id", -1),
                    "anomaly_score": score,
                }
            )

    anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return anomalies[:limit]


def get_stats(collection_name: str = "amazon_reviews") -> dict:
    """Get collection statistics."""
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    count = collection.count()
    result = collection.get(include=["metadatas"], limit=count)  # type: ignore[arg-type]

    metadatas = result["metadatas"] or []

    categories: dict[str, int] = {}
    quality_dist: dict[str, int] = {}
    cluster_ids: set[int] = set()
    anomaly_scores: list[float] = []

    for m in metadatas:
        cat = str(m.get("category", "unknown"))
        categories[cat] = categories.get(cat, 0) + 1

        qlabel = str(m.get("quality_label", "normal"))
        quality_dist[qlabel] = quality_dist.get(qlabel, 0) + 1

        cid = int(m.get("cluster_id", -1))
        cluster_ids.add(cid)

        anomaly_scores.append(float(m.get("anomaly_score", 0.0)))

    n_clusters = len([c for c in cluster_ids if c != -1])
    noise_count = sum(1 for c in cluster_ids if c == -1)
    anomaly_count = sum(1 for s in anomaly_scores if s > 0.7)

    return {
        "collection": collection_name,
        "total_reviews": count,
        "categories": categories,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "quality_distribution": quality_dist,
        "anomaly_count": anomaly_count,
        "anomaly_pct": (anomaly_count / count * 100) if count > 0 else 0,
    }
