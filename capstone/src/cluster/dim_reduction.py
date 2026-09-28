"""Dimensionality reduction to make embeddings plottable."""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

PROJECTIONS = ("pca", "umap", "none")


class DimReducer:
    """
    Projects high-dimensional vectors down to two dimensions.

    Embeddings have hundreds of components and cannot be plotted directly, so
    we compress them while preserving the neighbourhood structure as much as
    possible. That is what makes the cluster map readable.
    """

    def __init__(
        self, method: str = "pca", n_components: int = 2, random_state: int = 42
    ):
        normalized = method.strip().lower()
        if normalized not in PROJECTIONS:
            raise ValueError(
                f"Unknown projection '{method}'. Choose from {PROJECTIONS}."
            )
        self.method = normalized
        self.n_components = n_components
        self.random_state = random_state
        self._fitted = False
        self._variance_explained: Optional[float] = None

    def fit_transform(self, vectors: FloatArray) -> FloatArray:
        """Fit the projection on a matrix and return the 2D coordinates."""
        if vectors.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}")

        self._variance_explained = None

        if self.method == "none" or vectors.shape[1] <= self.n_components:
            # Already 2D (or close): return as is
            self._fitted = True
            if vectors.shape[1] == self.n_components:
                return np.asarray(vectors, dtype=np.float32)
            reduced = np.zeros((vectors.shape[0], self.n_components), dtype=np.float32)
            reduced[:, : vectors.shape[1]] = vectors
            return reduced

        if self.method == "pca":
            from sklearn.decomposition import PCA

            n_components = min(self.n_components, vectors.shape[0], vectors.shape[1])
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            coords = reducer.fit_transform(vectors)
            # A degenerate corpus (near-duplicate documents) makes scikit-learn
            # divide by a zero total variance, so the ratio comes back NaN.
            # NaN is not valid JSON, and it would poison analysis.json.
            ratio = float(reducer.explained_variance_ratio_.sum())
            self._variance_explained = ratio if math.isfinite(ratio) else None
        else:
            from umap import UMAP

            n_neighbors = max(2, min(15, vectors.shape[0] - 1))
            reducer = UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                random_state=self.random_state,
            )
            # A fixed seed is worth more than multithreading here: the cluster
            # map has to be reproducible across runs. UMAP warns that the seed
            # forces single-threaded layout, which is exactly what we asked for.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*n_jobs value.*overridden.*",
                    category=UserWarning,
                )
                coords = reducer.fit_transform(vectors)

        self._fitted = True
        return np.asarray(coords, dtype=np.float32)

    def transform(self, vectors: FloatArray) -> FloatArray:
        """Project new vectors using the already-fitted model."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        if self.method == "pca":
            raise RuntimeError("Reuse fit_transform() for new data with PCA")
        if self.method == "umap":
            from umap import UMAP

            # Re-fit is required for UMAP on new data in this implementation
            return self.fit_transform(vectors)
        return vectors

    @property
    def explains_variance(self) -> Optional[float]:
        """
        Share of the original variance kept by the projection.

        Only PCA reports this, and it is the honest way to say how much the 2D
        picture is a simplification: with 256 dimensions, two axes usually
        hold a small fraction of the information. UMAP preserves neighbourhoods
        rather than variance, so the number does not apply to it.
        """
        return self._variance_explained
