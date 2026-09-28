"""Embedding backend protocol and shared vector math."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@runtime_checkable
class EmbeddingModel(Protocol):
    """
    Turns text into a fixed-length list of numbers.

    Two texts that mean the same thing end up with vectors that are close to
    each other, which is what makes semantic search work.
    """

    @property
    def name(self) -> str:
        """Identifier of the backend, e.g. 'minilm' or 'tfidf'."""
        ...

    @property
    def dimension(self) -> int:
        """Length of the vectors produced by this backend."""
        ...

    def embed(self, texts: List[str]) -> FloatArray:
        """Embed a batch of texts into a (len(texts), dimension) matrix."""
        ...


def l2_normalize(matrix: FloatArray) -> FloatArray:
    """
    Scale every row to unit length.

    After normalization the dot product between two rows is already the cosine
    similarity, so searches only need a single matrix multiplication.
    """
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        if norm == 0.0:
            return np.asarray(matrix, dtype=np.float32)
        return np.asarray(matrix / norm, dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against zero-length rows (empty documents)
    norms[norms == 0.0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


def cosine_similarity(query: FloatArray, matrix: FloatArray) -> FloatArray:
    """
    Cosine similarity of one query vector against a matrix of vectors.

    A single stored vector may be passed as a 1D array, in which case the
    result is a matrix with one row.
    """
    query_flat = np.asarray(query, dtype=np.float32).reshape(-1)
    candidates = np.asarray(matrix, dtype=np.float32)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    query_norm = float(np.linalg.norm(query_flat))
    if query_norm == 0.0:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    return l2_normalize(candidates) @ (query_flat / query_norm).astype(np.float32)
