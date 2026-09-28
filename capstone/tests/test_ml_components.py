"""Tests for embeddings, clustering and quality scoring."""

from __future__ import annotations

import numpy as np
import pytest

from cluster.clustering import Clusterer
from cluster.dim_reduction import DimReducer
from embeddings.base import cosine_similarity, l2_normalize
from embeddings.factory import build_embedder
from embeddings.tfidf_svd import TfidfSvdEmbedder
from quality.anomalies import AnomalyDetector
from quality.classifier import QualityClassifier
from quality.features import extract_features, heuristic_quality_score

CORPUS = [
    "Neural networks learn representations from data using gradient descent and backpropagation.",
    "Deep learning models use convolutional layers to extract features from images.",
    "Convolutional neural networks achieve state of the art results on image classification.",
    "Machine learning models require large amounts of training data to generalize well.",
    "Gradient descent optimizes parameters by minimizing the loss function iteratively.",
    "Transfer learning reuses pretrained weights to speed up training on new tasks.",
]


def test_l2_normalize():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)


def test_cosine_similarity_identity():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matrix = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    scores = cosine_similarity(vector, matrix)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores[1] < scores[0]


def test_tfidf_embedder_fit_and_embed():
    embedder = TfidfSvdEmbedder(n_components=4)
    matrix = embedder.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert matrix.shape[1] == 4


def test_tfidf_embedder_requires_fit():
    embedder = TfidfSvdEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["some text"])


def test_tfidf_similar_texts_are_close():
    corpus = CORPUS * 4
    embedder = TfidfSvdEmbedder(n_components=8, min_df=1)
    vectors = embedder.fit_transform(corpus)
    assert vectors.shape[1] >= 4

    # Compare a network-training paper against every other document
    scores = cosine_similarity(vectors[0], vectors)
    own = scores[0]
    # A gradient-descent paper (index 4) must be closer than an image CNN paper (index 2)
    assert own == pytest.approx(1.0, abs=1e-4)
    assert scores[4] > scores[2]


def test_build_embedder_unknown():
    with pytest.raises(ValueError):
        build_embedder(backend="does-not-exist")


def test_dim_reducer_pca():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(30, 10)).astype(np.float32)
    reducer = DimReducer(method="pca", n_components=2)
    coords = reducer.fit_transform(vectors)
    assert coords.shape == (30, 2)


def test_clusterer_kmeans():
    rng = np.random.default_rng(1)
    vectors = np.vstack(
        [
            rng.normal(loc=0, scale=0.2, size=(20, 5)),
            rng.normal(loc=8, scale=0.2, size=(20, 5)),
        ]
    ).astype(np.float32)
    labels = Clusterer(algorithm="kmeans", n_clusters=2).fit_predict(vectors)
    assert int(np.unique(labels).size) == 2
    # The two blocks should land in different clusters
    assert labels[0] != labels[-1]


def test_anomaly_detector_flags_outlier():
    rng = np.random.default_rng(2)
    normal = rng.normal(size=(40, 4)).astype(np.float32)
    outlier = np.full((1, 4), 50.0, dtype=np.float32)
    vectors = np.vstack([normal, outlier])
    mask = AnomalyDetector(contamination=0.05).fit_predict(vectors)
    assert mask[-1]


def test_extract_features_shape():
    features = extract_features("Abstract. Method. Results. Conclusion.", "A Title")
    assert features["word_count"] > 0
    assert 0.0 <= features["section_coverage"] <= 1.0


def test_quality_score_prefers_structured_text():
    structured = (
        "Abstract This paper proposes a method. Introduction Prior work is limited. "
        "Method We design a network. Experiments We evaluate on three datasets. "
        "Results Our model improves accuracy. Conclusion We show the approach works."
    )
    noise = "aaaa " * 200
    assert heuristic_quality_score(extract_features(structured, "Good Paper")) > (
        heuristic_quality_score(extract_features(noise, "x"))
    )


def test_quality_classifier_labels():
    classifier = QualityClassifier()
    result = classifier.score_documents(
        CORPUS, [f"Title {i}" for i in range(len(CORPUS))]
    )
    assert len(result["scores"]) == len(CORPUS)
    assert all(label in {"high", "medium", "low"} for label in result["labels"])
