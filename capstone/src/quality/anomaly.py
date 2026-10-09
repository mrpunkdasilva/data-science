"""Anomaly detection using IsolationForest on embeddings."""

# mypy: ignore-errors

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from config import settings
from store.client import get_chroma_client, get_collection
from chromadb.api.models.Collection import Collection


def run_anomaly_detection(
    collection_name: str,
    contamination: float = 0.05,
) -> None:
    """
    Run IsolationForest anomaly detection on embeddings.

    Updates ChromaDB metadata with: anomaly_score (0-1, higher = more anomalous)
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    console_log("Fetching embeddings for anomaly detection...")
    count = collection.count()
    result = collection.get(include=["embeddings"], limit=count)  # type: ignore[arg-type]

    embeddings = np.array(result["embeddings"] or [])
    ids = result["ids"] or []

    console_log(f"Training IsolationForest on {len(embeddings)} embeddings...")

    iso_forest = IsolationForest(
        n_estimators=settings.ANOMALY_N_ESTIMATORS,
        contamination=contamination,
        random_state=settings.ANOMALY_RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(embeddings)

    # Get anomaly scores (negative = more anomalous)
    scores = -iso_forest.decision_function(embeddings)
    # Normalize to 0-1 range
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Save model
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_forest, settings.MODELS_DIR / "anomaly_detector.joblib")

    # Update ChromaDB in batches
    console_log("Updating ChromaDB with anomaly scores...")
    metadatas = [{"anomaly_score": float(s)} for s in scores]

    update_batch_size = 5000  # ChromaDB max batch size is 5461
    for offset in range(0, len(ids), update_batch_size):
        end = min(offset + update_batch_size, len(ids))
        collection.update(ids=ids[offset:end], metadatas=metadatas[offset:end])  # type: ignore[arg-type]
        console_log(
            f"  Updated batch {offset//update_batch_size + 1}/{(len(ids)-1)//update_batch_size + 1}"
        )

    n_anomalies = sum(1 for s in scores if s > 0.7)  # threshold for "high" anomaly
    console_log(
        f"Anomaly detection complete: {n_anomalies} high-anomaly verbetes (score > 0.7)"
    )


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[ANOMALY] {msg}")
