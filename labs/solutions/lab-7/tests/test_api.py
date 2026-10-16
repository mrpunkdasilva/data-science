"""Contract tests for the Wisconsin Cancer classification API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src import config
from src.app import app

client = TestClient(app)

_DATA = load_breast_cancer()
_FEATURE_NAMES = list(_DATA.feature_names)


def _sample_for(target: int) -> list[float]:
    index = next(i for i, value in enumerate(_DATA.target) if value == target)
    return [float(value) for value in _DATA.data[index]]


@pytest.fixture(scope="module")
def malignant_features() -> list[float]:
    return _sample_for(config.MALIGNANT_LABEL)


@pytest.fixture(scope="module")
def benign_features() -> list[float]:
    return _sample_for(config.BENIGN_LABEL)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Wisconsin Cancer Classification API"


def test_health_reports_loaded_model() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["n_features"] == config.N_FEATURES


def test_model_info_exposes_features_and_metrics() -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_features"] == config.N_FEATURES
    assert payload["feature_names"] == _FEATURE_NAMES
    assert payload["metrics"]


def test_predict_malignant_sample(malignant_features: list[float]) -> None:
    response = client.post("/predict", json={"features": malignant_features})
    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnosis"] == "malignant"
    assert payload["malignant_probability"] > 0.5


def test_predict_benign_sample(benign_features: list[float]) -> None:
    response = client.post("/predict", json={"features": benign_features})
    assert response.status_code == 200
    assert response.json()["diagnosis"] == "benign"


def test_predict_probabilities_are_complementary(
    malignant_features: list[float],
) -> None:
    payload = client.post("/predict", json={"features": malignant_features}).json()
    total = payload["malignant_probability"] + payload["benign_probability"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert payload["confidence"] == pytest.approx(
        max(payload["malignant_probability"], payload["benign_probability"])
    )


def test_predict_respects_latency_budget(malignant_features: list[float]) -> None:
    payload = client.post("/predict", json={"features": malignant_features}).json()
    assert payload["inference_ms"] < config.LATENCY_BUDGET_MS


def test_predict_rejects_wrong_feature_count() -> None:
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422


def test_predict_rejects_non_numeric_values() -> None:
    response = client.post(
        "/predict", json={"features": ["not-a-number"] * config.N_FEATURES}
    )
    assert response.status_code == 422


def test_batch_predict_matches_single_predictions(
    malignant_features: list[float], benign_features: list[float]
) -> None:
    payload = {
        "instances": [malignant_features, benign_features],
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["predictions"][0]["diagnosis"] == "malignant"
    assert body["predictions"][1]["diagnosis"] == "benign"
    assert body["inference_ms"] < config.LATENCY_BUDGET_MS


def test_batch_predict_rejects_bad_shape() -> None:
    response = client.post("/predict/batch", json={"instances": [[1.0, 2.0]]})
    assert response.status_code == 422
