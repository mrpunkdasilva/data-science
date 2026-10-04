"""Tests for capstone pipeline."""

import pytest
import numpy as np
from config import settings


def test_config_loads():
    """Test that settings load correctly."""
    assert settings.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.EMBEDDING_DIM == 384
    assert settings.COLLECTION_NAME == "amazon_reviews"
    assert settings.TARGET_SAMPLES == 15000


def test_embedding_dim():
    """Test embedding dimension matches config."""
    from embed.model import get_model

    model = get_model()
    emb = model.encode(["test"], normalize_embeddings=True)
    assert emb.shape == (1, settings.EMBEDDING_DIM)


def test_chroma_client():
    """Test ChromaDB client creation."""
    from store.client import get_chroma_client

    client = get_chroma_client()
    assert client is not None


def test_heuristic_labels():
    """Test heuristic quality labeling logic."""
    from ingest.loader import _heuristic_quality_label
    import pandas as pd

    # helpful: high helpful_vote, high rating, long text (no funny keywords)
    row = pd.Series(
        {
            "text": "This product is amazing and works great",
            "helpful_vote": 10,
            "rating": 5,
            "text_len": 150,
        }
    )
    assert _heuristic_quality_label(row) == "helpful"

    # funny: has funny keywords
    row2 = pd.Series(
        {
            "text": "This is so funny haha 😂",
            "helpful_vote": 5,
            "rating": 4,
            "text_len": 30,
        }
    )
    assert _heuristic_quality_label(row2) == "funny"

    # weird: low rating, high helpful, weird keywords
    row3 = pd.Series(
        {
            "text": "Weird strange bizarre product",
            "helpful_vote": 5,
            "rating": 1,
            "text_len": 40,
        }
    )
    assert _heuristic_quality_label(row3) == "weird"

    # fake_suspect: 5 stars, 0 helpful, short, marketing keywords
    row4 = pd.Series(
        {
            "text": "Best ever product highly recommend",
            "helpful_vote": 0,
            "rating": 5,
            "text_len": 50,
        }
    )
    assert _heuristic_quality_label(row4) == "fake_suspect"

    # normal: default
    row5 = pd.Series(
        {
            "text": "Average product nothing special",
            "helpful_vote": 1,
            "rating": 3,
            "text_len": 60,
        }
    )
    assert _heuristic_quality_label(row5) == "normal"


def test_isolation_forest():
    """Test anomaly detection basic functionality."""
    from sklearn.ensemble import IsolationForest

    X = np.random.randn(100, 384)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    # Use score_samples which returns negative log-likelihood (higher = more normal)
    # Negate to get anomaly score (higher = more anomalous)
    scores = -clf.score_samples(X)
    # Normalize to 0-1 range
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    assert len(scores) == 100
    assert scores.min() >= 0
    assert scores.max() <= 1


def test_logistic_regression():
    """Test classifier basic functionality."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["good product", "bad product", "great item", "terrible item"]
    y = ["positive", "negative", "positive", "negative"]
    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(X)
    clf = LogisticRegression(random_state=42)
    clf.fit(X_vec, y)
    pred = clf.predict(vec.transform(["awesome product"]))
    assert pred[0] == "positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
