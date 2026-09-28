"""Semantic search on top of the local vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from embeddings.base import EmbeddingModel
from embeddings.factory import build_embedder
from index.vector_store import SearchHit, VectorStore

TFIDF_MODEL_FILE = "tfidf_svd.joblib"


def load_embedder(backend: str, model_dir: Path) -> EmbeddingModel:
    """
    Build the embedder used to project queries.

    Backends that need fitting on the corpus reload their persisted model,
    because a query must be projected into the very same space as the index.
    """
    if backend.strip().lower() == "tfidf":
        from embeddings.tfidf_svd import TfidfSvdEmbedder

        model_path = Path(model_dir) / "models" / TFIDF_MODEL_FILE
        if not model_path.exists():
            model_path = Path(model_dir) / TFIDF_MODEL_FILE
        if model_path.exists():
            return TfidfSvdEmbedder.load(model_path)
        raise FileNotFoundError(
            f"No fitted TF-IDF model at {model_path}. Rebuild the index with "
            "'capstone build --backend tfidf'."
        )

    return build_embedder(backend=backend)


class SemanticSearch:
    """Turns a text query into ranked documents using a vector store."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Optional[EmbeddingModel] = None,
        backend: Optional[str] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.store = store
        self._embedder = embedder
        self._backend = backend or store.backend
        self._model_dir = model_dir or store.path.parent

    @property
    def embedder(self) -> EmbeddingModel:
        """
        Build the query embedder lazily.

        For TF-IDF the fitted model saved during the build is reused, so the
        query is projected with the same vocabulary as the corpus.
        """
        if self._embedder is None:
            self._embedder = load_embedder(self._backend, self._model_dir)
        return self._embedder

    def query(self, text: str, k: int = 5) -> List[SearchHit]:
        """Search for the documents closest to a natural language query."""
        vector = self.embedder.embed([text])[0]
        return self.store.search(vector, k=k)

    def query_with_filters(
        self, text: str, k: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """Search restricted to documents matching a metadata filter."""
        vector = self.embedder.embed([text])[0]
        return self.store.search(vector, k=k, where=where)

    def similar(self, doc_id: str, k: int = 5) -> List[SearchHit]:
        """Find documents similar to an indexed one (more-like-this)."""
        if not self.store.contains(doc_id):
            raise KeyError(f"Document '{doc_id}' is not in the index")

        hits = self.store.search(self.store.get_vector(doc_id), k=k + 1)
        return [hit for hit in hits if hit.doc_id != doc_id][:k]
