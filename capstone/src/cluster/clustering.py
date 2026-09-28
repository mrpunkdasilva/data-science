"""Unsupervised clustering of document embeddings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

ALGORITHMS = ("kmeans", "hdbscan", "agglomerative")


class Clusterer:
    """
    Groups documents into clusters without any labels.

    KMeans is the default because it is fast, deterministic and easy to explain
    in a presentation. HDBSCAN is available for corpora where the number of
    groups is not known upfront, since it can also flag noise points.
    """

    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: int = 8,
        random_state: int = 42,
    ) -> None:
        normalized = algorithm.strip().lower()
        if normalized not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from {ALGORITHMS}."
            )
        self.algorithm = normalized
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model: Any = None
        self.labels_: Optional[NDArray[np.int64]] = None
        self.probabilities_: Optional[FloatArray] = None
        self.outliers_: Optional[NDArray[np.bool_]] = None

    def fit_predict(self, vectors: FloatArray) -> NDArray[np.int64]:
        """Cluster the vectors and return one label per document."""
        if vectors.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}")

        if self.algorithm == "kmeans":
            from sklearn.cluster import KMeans

            k = min(self.n_clusters, vectors.shape[0])
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(vectors)
            # Distance to the closest centroid, used as a confidence proxy
            distances = model.transform(vectors)
            confidence = 1.0 - (distances.min(axis=1) / (distances.max() + 1e-9))
            self.probabilities_ = np.asarray(confidence, dtype=np.float32)

        elif self.algorithm == "hdbscan":
            from sklearn.cluster import HDBSCAN

            model = HDBSCAN(min_cluster_size=15)
            labels = model.fit_predict(vectors)
            self.probabilities_ = np.asarray(
                getattr(model, "probabilities_", np.zeros(len(labels))),
                dtype=np.float32,
            )
            self.outliers_ = np.asarray(labels == -1, dtype=bool)
            # HDBSCAN uses -1 for noise; shift labels so they are always >= 0
            if self.outliers_.any():
                unique = {
                    label: index
                    for index, label in enumerate(sorted(set(labels) - {-1}))
                }
                labels = np.array(
                    [unique.get(int(label), -1) for label in labels], dtype=np.int64
                )

        else:
            from sklearn.cluster import AgglomerativeClustering

            k = min(self.n_clusters, vectors.shape[0])
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(vectors)

        self._model = model
        self.labels_ = np.asarray(labels, dtype=np.int64)
        return self.labels_

    def cluster_profiles(self, vectors: FloatArray) -> Dict[int, Dict[str, Any]]:
        """Describe each cluster with its size and centroid."""
        if self.labels_ is None:
            raise RuntimeError("Call fit_predict() first")

        profiles: Dict[int, Dict[str, Any]] = {}
        for label in sorted({int(value) for value in self.labels_}):
            mask = self.labels_ == label
            member_vectors = vectors[mask]
            centroid = member_vectors.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            profiles[label] = {
                "size": int(mask.sum()),
                "centroid": centroid.astype(np.float32),
            }
        return profiles

    def find_optimal_k(
        self, vectors: FloatArray, max_k: int = 15
    ) -> List[Dict[str, float]]:
        """
        Elbow analysis: inertia for k=2..max_k, to help justify the cluster count.
        """
        from sklearn.cluster import KMeans

        scores: List[Dict[str, float]] = []
        upper = min(max_k, vectors.shape[0])
        for k in range(2, upper + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(vectors)
            scores.append({"k": float(k), "inertia": float(model.inertia_)})
        return scores
