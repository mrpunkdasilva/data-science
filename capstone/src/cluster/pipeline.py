"""Clustering pipeline: UMAP dimensionality reduction + HDBSCAN clustering."""

# mypy: ignore-errors

import numpy as np
import umap
import hdbscan
from config import settings
from store.client import get_chroma_client, get_collection
from chromadb.api.models.Collection import Collection


def run_clustering(
    collection_name: str,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 2,
    umap_metric: str = "cosine",
    umap_random_state: int = 42,
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int = 5,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_epsilon: float = 0.0,
) -> None:
    """
    Run UMAP + HDBSCAN clustering on stored embeddings.

    Updates ChromaDB metadata with: cluster_id, umap_x, umap_y
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    console_log(f"Fetching embeddings from {collection_name}...")
    count = collection.count()
    if count == 0:
        console_log("Collection is empty!")
        return

    # Fetch all embeddings in batches
    all_embeddings: list[list[float]] = []
    all_ids: list[str] = []
    batch_size = 1000
    for offset in range(0, count, batch_size):
        result = collection.get(
            include=["embeddings"],  # type: ignore[arg-type]
            limit=min(batch_size, count - offset),
            offset=offset,
        )
        all_embeddings.extend(result["embeddings"] or [])
        all_ids.extend(result["ids"] or [])

    embeddings = np.array(all_embeddings)
    console_log(f"Running UMAP on {len(embeddings)} embeddings...")

    # UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=umap_n_components,
        metric=umap_metric,
        random_state=umap_random_state,
        verbose=True,
    )
    umap_embeddings = reducer.fit_transform(embeddings)

    console_log("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Update ChromaDB with cluster info
    console_log("Updating ChromaDB with cluster labels and UMAP coordinates...")
    metadatas: list[dict[str, float | int]] = []
    for i, (uid, label) in enumerate(zip(all_ids, cluster_labels)):
        metadatas.append(
            {
                "cluster_id": int(label),
                "umap_x": float(umap_embeddings[i, 0]),
                "umap_y": float(umap_embeddings[i, 1]),
            }
        )

    collection.update(ids=all_ids, metadatas=metadatas)  # type: ignore[arg-type]

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = sum(1 for l in cluster_labels if l == -1)
    console_log(f"Clustering complete: {n_clusters} clusters, {noise_count} noise points")


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[CLUSTER] {msg}")
