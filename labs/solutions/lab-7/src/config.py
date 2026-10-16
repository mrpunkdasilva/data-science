"""Shared configuration and constants for the Lab 7 inference API."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "cancer_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODEL_VERSION = "1.0.0"

N_FEATURES = 30

MALIGNANT_LABEL = 0
BENIGN_LABEL = 1
CLASS_LABELS = {MALIGNANT_LABEL: "malignant", BENIGN_LABEL: "benign"}

LATENCY_BUDGET_MS = 100.0

__all__ = [
    "BASE_DIR",
    "MODELS_DIR",
    "DATA_DIR",
    "MODEL_PATH",
    "METADATA_PATH",
    "MODEL_VERSION",
    "N_FEATURES",
    "MALIGNANT_LABEL",
    "BENIGN_LABEL",
    "CLASS_LABELS",
    "LATENCY_BUDGET_MS",
]
