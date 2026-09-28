"""Factory to build the embedding backend chosen by configuration."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import EmbeddingModel, FloatArray
from .tfidf_svd import TfidfSvdEmbedder

BACKENDS = ("tfidf", "minilm")

DEFAULT_BACKEND = "tfidf"


def build_embedder(
    backend: str = DEFAULT_BACKEND,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> EmbeddingModel:
    """
    Instantiate an embedding backend by name.

    'minilm' needs sentence-transformers and a one-time model download, while
    'tfidf' only needs scikit-learn and works offline.
    """
    normalized = backend.strip().lower()

    if normalized == "tfidf":
        return TfidfSvdEmbedder()

    if normalized == "minilm":
        from .sbert import DEFAULT_MODEL, SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=model_name or DEFAULT_MODEL,
            batch_size=batch_size,
        )

    raise ValueError(f"Unknown embedding backend '{backend}'. Choose from {BACKENDS}.")


def embed_corpus(
    texts: List[str],
    backend: str = DEFAULT_BACKEND,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> tuple[FloatArray, str, int]:
    """
    Embed a whole corpus, fitting the backend when it needs fitting.

    Returns the matrix, the backend name and the vector dimension.
    """
    if not texts:
        raise ValueError("Cannot embed an empty corpus")

    model = build_embedder(
        backend=backend, model_name=model_name, batch_size=batch_size
    )

    if isinstance(model, TfidfSvdEmbedder):
        matrix = model.fit_transform(texts)
    else:
        matrix = model.embed(texts)

    if matrix.shape[0] != len(texts):
        raise RuntimeError(
            f"Embedding backend returned {matrix.shape[0]} vectors for {len(texts)} texts"
        )

    return np.asarray(matrix, dtype=np.float32), model.name, model.dimension
