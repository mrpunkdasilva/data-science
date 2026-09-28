"""Sentence-Transformers backend (all-MiniLM-L6-v2)."""

from __future__ import annotations

import os
from typing import List

import numpy as np

from .base import FloatArray

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """
    Neural embeddings from a local Sentence-Transformers model.

    The model understands meaning, so a search for "how do I train a network"
    also finds documents about "optimizing artificial models" even without
    shared words.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        cache_folder: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install -e '.[embeddings]'"
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        self._cache_folder = cache_folder or os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME"
        )
        self._model = SentenceTransformer(model_name, cache_folder=self._cache_folder)
        reported = self._model.get_sentence_embedding_dimension()
        if reported is None:
            raise RuntimeError(
                f"Model '{model_name}' did not report an embedding dimension"
            )
        self._dimension = int(reported)

    @property
    def name(self) -> str:
        return "minilm"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
