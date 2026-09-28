"""Quality scoring and classification of documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .features import (
    FEATURE_NAMES,
    extract_features,
    features_to_matrix,
    heuristic_quality_score,
)

LABELS = ("high", "medium", "low")
THRESHOLDS = {"high": 0.66, "medium": 0.33}


class QualityClassifier:
    """
    Scores and classifies documents on quality.

    The default scorer is a transparent rule set (no labels needed). Pass
    pseudo-labels to train() to switch to a supervised Random Forest, which is
    the "learned" quality model the project asks for.
    """

    def __init__(self, use_learned_model: bool = False, random_state: int = 42) -> None:
        self.use_learned_model = use_learned_model
        self.random_state = random_state
        self._model: Any = None
        self._feature_names = FEATURE_NAMES

    def score_documents(
        self, texts: List[str], titles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute quality features, scores and labels for a list of documents."""
        if not texts:
            raise ValueError("Cannot score an empty document list")

        titles = titles or [""] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")

        features = [extract_features(text, title) for text, title in zip(texts, titles)]
        matrix = features_to_matrix(features)

        if self.use_learned_model and self._model is not None:
            scores = np.asarray(
                self._model.predict_proba(matrix)[:, 1], dtype=np.float32
            )
        else:
            scores = np.array(
                [heuristic_quality_score(record) for record in features],
                dtype=np.float32,
            )

        labels = [self.to_label(float(score)) for score in scores]
        return {
            "scores": scores,
            "labels": labels,
            "features": features,
            "matrix": matrix,
        }

    def train(
        self, texts: List[str], labels: List[str], titles: Optional[List[str]] = None
    ) -> None:
        """Train a supervised classifier from pseudo-labels."""
        from sklearn.ensemble import RandomForestClassifier

        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")

        titles = titles or [""] * len(texts)
        features = [extract_features(text, title) for text, title in zip(texts, titles)]
        matrix = features_to_matrix(features)

        numeric = np.array(
            [{"high": 2, "medium": 1, "low": 0}[label] for label in labels],
            dtype=np.int64,
        )
        self._model = RandomForestClassifier(
            n_estimators=120, random_state=self.random_state, oob_score=True
        )
        self._model.fit(matrix, numeric)
        self.use_learned_model = True

    @staticmethod
    def to_label(score: float) -> str:
        """Map a 0-1 score to a high/medium/low label."""
        if score >= THRESHOLDS["high"]:
            return "high"
        if score >= THRESHOLDS["medium"]:
            return "medium"
        return "low"

    def feature_importances(self) -> Optional[List[Tuple[str, float]]]:
        """Which features matter most in the learned model."""
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_names, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(name, float(value)) for name, value in pairs]

    def summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate quality distribution for the CLI and the UI."""
        labels: List[str] = result["labels"]
        scores: NDArray[np.float32] = result["scores"]
        distribution = {label: labels.count(label) for label in LABELS}
        return {
            "documents": len(labels),
            "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            "median_score": float(np.median(scores)) if len(scores) else 0.0,
            "distribution": distribution,
        }
