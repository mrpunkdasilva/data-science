"""Loads the persisted model and exposes fast single/batch inference helpers."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any, Sequence

import joblib
import numpy as np
import numpy.typing as npt

from src import config

FeatureVector = Sequence[float]
FeatureMatrix = Sequence[Sequence[float]]


class ModelService:
    """Wraps the persisted scikit-learn pipeline with prediction helpers."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self._model = model
        self._metadata = metadata
        self._feature_names = list(metadata["feature_names"])
        self._malignant_index = list(model.classes_).index(config.MALIGNANT_LABEL)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("model_version", config.MODEL_VERSION))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _as_matrix(
        self, features: FeatureVector | FeatureMatrix
    ) -> npt.NDArray[np.float64]:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features per sample, "
                f"received shape {array.shape}."
            )
        return array

    def predict_one(self, features: FeatureVector) -> dict[str, Any]:
        """Classify a single sample and report the server-side latency."""
        matrix = self._as_matrix(features)
        start = time.perf_counter()
        probabilities = self._model.predict_proba(matrix)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        malignant_probability = float(probabilities[0][self._malignant_index])
        benign_probability = float(1.0 - malignant_probability)
        label = (
            config.MALIGNANT_LABEL
            if malignant_probability >= 0.5
            else config.BENIGN_LABEL
        )
        return {
            "diagnosis": config.CLASS_LABELS[label],
            "malignant_probability": malignant_probability,
            "benign_probability": benign_probability,
            "confidence": max(malignant_probability, benign_probability),
            "inference_ms": elapsed_ms,
            "model_version": self.model_version,
        }

    def predict_batch(
        self, instances: Sequence[FeatureVector]
    ) -> tuple[list[dict[str, Any]], float]:
        """Classify many samples in one vectorised call."""
        matrix = self._as_matrix(instances)
        start = time.perf_counter()
        probabilities = self._model.predict_proba(matrix)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        malignant = probabilities[:, self._malignant_index]
        results: list[dict[str, Any]] = []
        for probability in malignant:
            malignant_probability = float(probability)
            benign_probability = float(1.0 - malignant_probability)
            label = (
                config.MALIGNANT_LABEL
                if malignant_probability >= 0.5
                else config.BENIGN_LABEL
            )
            results.append(
                {
                    "diagnosis": config.CLASS_LABELS[label],
                    "malignant_probability": malignant_probability,
                    "benign_probability": benign_probability,
                    "confidence": max(malignant_probability, benign_probability),
                    "inference_ms": elapsed_ms / len(malignant),
                    "model_version": self.model_version,
                }
            )
        return results, elapsed_ms


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    """Return the cached singleton model service."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {config.MODEL_PATH}. "
            "Run the lab-7 notebook to train and persist the model."
        )
    model = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    return ModelService(model, metadata)


def clear_service_cache() -> None:
    """Clear the cached model service (used by tests)."""
    get_service.cache_clear()


__all__ = ["ModelService", "get_service", "clear_service_cache"]
