"""FastAPI service that classifies Wisconsin cancer tissue samples."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException

from src import config
from src.model import get_service
from src.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    Prediction,
)

app = FastAPI(
    title="Wisconsin Cancer Classification API",
    version=config.MODEL_VERSION,
    description=(
        "End-to-end inference service for the Wisconsin Diagnostic Cancer "
        "dataset. Submit the 30 image-derived features of a tissue sample and "
        "receive a malignant/benign diagnosis with a calibrated probability."
    ),
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Redirect-friendly landing payload."""
    return {
        "service": "Wisconsin Cancer Classification API",
        "version": config.MODEL_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report whether the model is loaded and ready to serve."""
    try:
        service = get_service()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=service.model_version,
        n_features=service.n_features,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Expose the persisted model metadata and evaluation metrics."""
    try:
        service = get_service()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    metadata = service.metadata
    return ModelInfoResponse(
        model_version=service.model_version,
        algorithm=str(metadata.get("algorithm", "unknown")),
        trained_on=str(metadata.get("trained_on", "unknown")),
        n_features=service.n_features,
        feature_names=service.feature_names,
        metrics={
            key: float(value)
            for key, value in metadata.get("metrics", {}).items()
            if isinstance(value, (int, float))
        },
    )


def _to_prediction(payload: dict[str, Any]) -> Prediction:
    return Prediction(**payload)


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(request: PredictRequest) -> Prediction:
    """Classify a single tissue sample."""
    service = get_service()
    try:
        result = service.predict_one(request.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _to_prediction(result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Classify a batch of tissue samples in one pass."""
    service = get_service()
    start = time.perf_counter()
    try:
        results, inference_ms = service.predict_batch(request.instances)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    total_ms = (time.perf_counter() - start) * 1000.0
    return BatchPredictResponse(
        predictions=[_to_prediction(result) for result in results],
        count=len(results),
        inference_ms=total_ms if total_ms >= inference_ms else inference_ms,
        model_version=service.model_version,
    )


__all__ = ["app"]
