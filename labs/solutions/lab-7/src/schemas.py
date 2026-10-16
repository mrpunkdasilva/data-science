"""Pydantic request/response models for the classification API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.config import JsonValue

from src import config


class _Schema(BaseModel):
    """Base model that allows ``model_*`` field names."""

    model_config = ConfigDict(protected_namespaces=())


_EXAMPLE_FEATURES: list[JsonValue] = [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]


class PredictRequest(_Schema):
    """A single tissue sample described by its 30 image-derived features."""

    features: list[float] = Field(
        ...,
        description=(
            "The 30 numeric features computed from the cell nuclei image, "
            "in the canonical Wisconsin Diagnostic order."
        ),
    )

    model_config = {
        "json_schema_extra": {"examples": [{"features": _EXAMPLE_FEATURES}]}
    }

    @field_validator("features")
    @classmethod
    def _check_length(cls, value: list[float]) -> list[float]:
        if len(value) != config.N_FEATURES:
            raise ValueError(
                f"Expected {config.N_FEATURES} features, received {len(value)}."
            )
        return value


class BatchPredictRequest(_Schema):
    """A batch of tissue samples."""

    instances: list[list[float]] = Field(
        ...,
        min_length=1,
        description="A list of feature vectors, each with 30 values.",
    )

    model_config = {
        "json_schema_extra": {"examples": [{"instances": [_EXAMPLE_FEATURES]}]}
    }

    @field_validator("instances")
    @classmethod
    def _check_shape(cls, value: list[list[float]]) -> list[list[float]]:
        for index, features in enumerate(value):
            if len(features) != config.N_FEATURES:
                raise ValueError(
                    f"Instance {index} has {len(features)} features; "
                    f"expected {config.N_FEATURES}."
                )
        return value


class Prediction(_Schema):
    """The classification result for a single tissue sample."""

    diagnosis: str = Field(..., description="'malignant' or 'benign'.")
    malignant_probability: float = Field(..., ge=0.0, le=1.0)
    benign_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_ms: float = Field(..., description="Server-side inference time.")
    model_version: str


class BatchPredictResponse(_Schema):
    """Classification results for a batch of tissue samples."""

    predictions: list[Prediction]
    count: int
    inference_ms: float
    model_version: str


class HealthResponse(_Schema):
    """Service health information."""

    status: str
    model_loaded: bool
    model_version: str
    n_features: int


class ModelInfoResponse(_Schema):
    """Metadata about the deployed model."""

    model_version: str
    algorithm: str
    trained_on: str
    n_features: int
    feature_names: list[str]
    metrics: dict[str, float]


__all__ = [
    "PredictRequest",
    "BatchPredictRequest",
    "Prediction",
    "BatchPredictResponse",
    "HealthResponse",
    "ModelInfoResponse",
]
