"""Anomaly detection over document embeddings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

DETECTORS = ("isolation_forest", "knn_distance")


class AnomalyDetector:
    """
    Finds documents that do not fit the corpus.

    A paper whose embedding is far away from every other one is either off
    topic, broken, or a duplicate-ish outlier. Isolation Forest measures that
    directly: points that are easy to isolate are rare, and rare means
    anomalous.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        normalized = method.strip().lower()
        if normalized not in DETECTORS:
            raise ValueError(f"Unknown detector '{method}'. Choose from {DETECTORS}.")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.method = normalized
        self.contamination = contamination
        self.random_state = random_state
        self._model: Any = None
        self.is_anomaly_: Optional[NDArray[np.bool_]] = None
        self.scores_: Optional[FloatArray] = None

    def fit_predict(self, vectors: FloatArray) -> NDArray[np.bool_]:
        """Flag anomalous documents, returning a boolean mask."""
        if vectors.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}")
        if vectors.shape[0] < 5:
            raise ValueError("Need at least 5 documents to detect anomalies")

        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
            )
            model.fit(vectors)
            # score_samples: higher means more normal, so we invert the sign
            scores = np.asarray(-model.score_samples(vectors), dtype=np.float32)
            mask = np.asarray(model.predict(vectors) == -1, dtype=bool)
        else:
            from sklearn.neighbors import NearestNeighbors

            k = max(2, min(5, vectors.shape[0] - 1))
            model = NearestNeighbors(n_neighbors=k).fit(vectors)
            distances, _ = model.kneighbors(vectors)
            mean_distance = distances[:, 1:].mean(axis=1)
            scores = np.asarray(mean_distance, dtype=np.float32)
            threshold = float(
                np.percentile(mean_distance, 100 * (1 - self.contamination))
            )
            mask = np.asarray(mean_distance > threshold, dtype=bool)

        self._model = model
        self.scores_ = scores
        self.is_anomaly_ = mask
        return mask

    def summary(self, doc_ids: Tuple[str, ...] | list[str]) -> Dict[str, Any]:
        """Count of anomalies and the worst offenders."""
        if self.is_anomaly_ is None or self.scores_ is None:
            raise RuntimeError("Call fit_predict() first")

        scores = self.scores_
        mask = self.is_anomaly_
        flagged: List[Dict[str, Any]] = [
            {"doc_id": doc_id, "score": float(scores[i])}
            for i, doc_id in enumerate(doc_ids)
            if bool(mask[i])
        ]
        flagged.sort(key=lambda item: float(item["score"]), reverse=True)
        return {
            "total": len(tuple(doc_ids)),
            "anomalies": len(flagged),
            "rate": len(flagged) / len(tuple(doc_ids)) if tuple(doc_ids) else 0.0,
            "flagged": flagged[:20],
        }
