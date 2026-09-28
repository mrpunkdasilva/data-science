"""TF-IDF + SVD backend, fully local and dependency-free."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import FloatArray


class TfidfSvdEmbedder:
    """
    Lexical embeddings built with scikit-learn, no model download required.

    It works by counting how often each word appears, weighting rare words
    higher, and compressing the result with truncated SVD. It only matches
    shared words, so it is weaker than a neural model, but it always works
    offline and keeps CI fast.
    """

    def __init__(
        self,
        n_components: int = 256,
        max_features: int = 20000,
        stop_words: str = "english",
        min_df: int = 2,
    ) -> None:
        self._n_components = n_components
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._svd: Optional[TruncatedSVD] = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "tfidf"

    @property
    def dimension(self) -> int:
        if self._is_fitted and self._svd is not None:
            return int(self._svd.n_components)
        return self._n_components

    def fit(self, texts: List[str]) -> None:
        """Learn the vocabulary and the SVD projection from a corpus."""
        if not texts:
            raise ValueError("Cannot fit the embedder on an empty corpus")

        matrix = self._vectorizer.fit_transform(texts)
        n_components = min(self._n_components, max(matrix.shape[1] - 1, 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)
        self._is_fitted = True

    def embed(self, texts: List[str]) -> FloatArray:
        if not self._is_fitted or self._svd is None:
            raise RuntimeError("Call fit() with a corpus before embedding texts")

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        matrix = self._vectorizer.transform(texts)
        reduced = self._svd.transform(matrix)
        return np.asarray(normalize(reduced), dtype=np.float32)

    def fit_transform(self, texts: List[str]) -> FloatArray:
        """Fit on a corpus and return the embeddings of that same corpus."""
        self.fit(texts)
        return self.embed(texts)

    def transform_query(self, text: str) -> FloatArray:
        """Embed a single query string, used by semantic search."""
        return np.asarray(self.embed([text])[0], dtype=np.float32)

    def save(self, path: Path) -> None:
        """
        Persist the fitted vocabulary and projection.

        This is required for search: a new query has to be projected with the
        exact same vocabulary and SVD basis learned from the corpus, otherwise
        the query would land in a different space than the indexed documents.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before saving the model")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "n_components": self._n_components,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        """Restore a model saved with save()."""
        if not path.exists():
            raise FileNotFoundError(
                f"No TF-IDF model found at {path}. Run 'capstone build' first."
            )

        payload = joblib.load(path)
        embedder = cls(n_components=int(payload.get("n_components", 256)))
        embedder._vectorizer = payload["vectorizer"]
        embedder._svd = payload["svd"]
        embedder._is_fitted = True
        return embedder
