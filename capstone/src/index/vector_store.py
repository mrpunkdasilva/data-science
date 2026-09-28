"""Hand-rolled vector store backed by NumPy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from embeddings.base import FloatArray, l2_normalize

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"


@dataclass
class SearchHit:
    """One document returned by a similarity search."""

    doc_id: str
    score: float
    metadata: Dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", self.doc_id))


class VectorStore:
    """
    Minimal local vector database.

    Vectors are stored L2-normalized in a single .npy file, so a cosine
    similarity search is one matrix multiplication. Metadata is kept in a JSON
    file, one record per document, so it stays readable and diffable in git.
    """

    def __init__(self, path: Path, dimension: int = 0, backend: str = "tfidf") -> None:
        self.path = Path(path)
        self.dimension = dimension
        self.backend = backend
        self._vectors: Optional[FloatArray] = None
        self._ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._position: Dict[str, int] = {}
        self._loaded = False

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def doc_ids(self) -> List[str]:
        return list(self._ids)

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return [self._metadata[doc_id] for doc_id in self._ids]

    @property
    def vectors(self) -> FloatArray:
        if self._vectors is None:
            raise RuntimeError("Index is empty, run build/embed first")
        return self._vectors

    def build(
        self,
        doc_ids: Sequence[str],
        vectors: FloatArray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Create the index from embeddings, normalizing them in place."""
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(doc_ids)} ids for {vectors.shape[0]} vectors, they must match"
            )
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Document ids must be unique")
        if metadata is not None and len(metadata) != len(doc_ids):
            raise ValueError("Metadata length must match the number of documents")

        if vectors.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}")

        self._vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self.dimension = int(self._vectors.shape[1])
        self._ids = list(doc_ids)
        self._metadata = {
            doc_id: dict(meta or {})
            for doc_id, meta in zip(doc_ids, metadata or [{}] * len(doc_ids))
        }
        self._position = {doc_id: i for i, doc_id in enumerate(self._ids)}
        self._loaded = True

    def add(
        self,
        doc_ids: Sequence[str],
        vectors: FloatArray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Append documents to an existing index."""
        if self._vectors is None:
            self.build(doc_ids, vectors, metadata)
            return

        new_vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if new_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.dimension}, got {new_vectors.shape[1]}"
            )

        fresh = [doc_id for doc_id in doc_ids if doc_id not in self._position]
        if len(fresh) != len(doc_ids):
            raise ValueError("Some document ids are already present in the index")
        if metadata is not None and len(metadata) != len(doc_ids):
            raise ValueError("Metadata length must match the number of documents")

        self._vectors = np.vstack([self._vectors, new_vectors])
        for offset, doc_id in enumerate(doc_ids):
            self._position[doc_id] = len(self._ids) + offset
            self._ids.append(doc_id)
        if metadata is not None:
            for doc_id, meta in zip(doc_ids, metadata):
                self._metadata[doc_id] = dict(meta)

    def delete(self, doc_id: str) -> bool:
        """Remove one document, rebuilding the matrix when needed."""
        if doc_id not in self._position or self._vectors is None:
            return False

        position = self._position[doc_id]
        mask = np.ones(len(self._ids), dtype=bool)
        mask[position] = False
        self._vectors = self._vectors[mask]

        self._ids = [known_id for known_id in self._ids if known_id != doc_id]
        self._metadata.pop(doc_id, None)
        self._position = {known_id: i for i, known_id in enumerate(self._ids)}
        return True

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(doc_id)

    def contains(self, doc_id: str) -> bool:
        """Whether a document id is present in the index."""
        return doc_id in self._position

    def get_vector(self, doc_id: str) -> FloatArray:
        """Return the stored vector of one document, for more-like-this search."""
        position = self._position.get(doc_id)
        if position is None:
            raise KeyError(f"Document '{doc_id}' is not in the index")
        return np.asarray(self.vectors[position], dtype=np.float32)

    def search(
        self,
        query_vector: FloatArray,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """
        Return the k most similar documents to a query vector.

        Because the stored vectors are normalized, the dot product already is
        the cosine similarity.
        """
        if self._vectors is None or not self._ids:
            raise RuntimeError("Index is empty, run build/embed first")

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query has dimension {query.shape[0]}, but the index was built "
                f"with {self.dimension}-dimensional '{self.backend}' embeddings. "
                "A query must use the same backend as the index: rebuild the "
                "index or pass the matching --backend."
            )

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise ValueError(
                "The query produced a zero vector, which means none of its terms "
                "appear in the indexed vocabulary. Try words that occur in the "
                "corpus, or rebuild the index with a richer embedding backend."
            )
        scores = self._vectors @ (query / norm)

        if where:
            mask = np.array(
                [_matches(self._metadata[doc], where) for doc in self._ids], dtype=bool
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(k, len(self._ids))
        # argpartition finds the k best in linear time, then we sort just those
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            SearchHit(
                doc_id=self._ids[int(position)],
                score=float(scores[int(position)]),
                metadata=dict(self._metadata[self._ids[int(position)]]),
            )
            for position in top
            if np.isfinite(scores[int(position)])
        ]

    def save(self) -> None:
        """Persist the index to disk."""
        if self._vectors is None:
            raise RuntimeError("Nothing to save, the index is empty")

        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / EMBEDDINGS_FILE, self._vectors)

        with (self.path / METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(self._ids, handle)

        records = [self._metadata[doc_id] for doc_id in self._ids]
        with (self.path / f"{METADATA_FILE}.records").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)

        with (self.path / MANIFEST_FILE).open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "count": len(self._ids),
                    "dimension": self.dimension,
                    "backend": self.backend,
                    "normalized": True,
                },
                handle,
                indent=1,
            )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a previously saved index."""
        path = Path(path)
        embeddings_path = path / EMBEDDINGS_FILE
        records_path = path / f"{METADATA_FILE}.records"

        if not embeddings_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                f"No index found at {path}. Run 'capstone build' first."
            )

        vectors = np.load(embeddings_path).astype(np.float32)
        with (path / METADATA_FILE).open(encoding="utf-8") as handle:
            doc_ids: List[str] = json.load(handle)
        with records_path.open(encoding="utf-8") as handle:
            records: List[Dict[str, Any]] = json.load(handle)

        backend = "tfidf"
        manifest_path = path / MANIFEST_FILE
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                backend = str(json.load(handle).get("backend", "tfidf"))

        store = cls(path=path, dimension=int(vectors.shape[1]), backend=backend)
        store._vectors = vectors
        store._ids = doc_ids
        store._metadata = {doc_id: dict(rec) for doc_id, rec in zip(doc_ids, records)}
        store._position = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        store._loaded = True
        return store

    def stats(self) -> Dict[str, Any]:
        """Summary of the index, shown by the CLI and the UI."""
        return {
            "documents": len(self._ids),
            "dimension": self.dimension,
            "backend": self.backend,
            "path": str(self.path),
        }


def _matches(metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """Evaluate a simple equality filter over document metadata."""
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def iter_documents(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Small helper used by the CLI to iterate metadata records."""
    return list(records)
