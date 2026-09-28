"""Tests for the vector store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from index.vector_store import VectorStore


def make_vectors(n: int = 10, dim: int = 8, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_build_and_search(tmp_path: Path) -> None:
    vectors = make_vectors()
    store = VectorStore(path=tmp_path, dimension=8)
    store.build([f"doc{i}" for i in range(10)], vectors)

    hits = store.search(vectors[3], k=3)
    assert len(hits) == 3
    # The exact document should rank first with similarity ~1
    assert hits[0].doc_id == "doc3"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_normalization() -> None:
    vectors = make_vectors() * 100  # large magnitudes
    store = VectorStore(path=Path("/tmp/test_store_norm"), dimension=8)
    store.build([f"doc{i}" for i in range(10)], vectors)
    norms = np.linalg.norm(store.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_and_load(tmp_path: Path) -> None:
    vectors = make_vectors()
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(
        [f"doc{i}" for i in range(10)],
        vectors,
        [{"title": f"Doc {i}"} for i in range(10)],
    )
    store.save()

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 10
    assert loaded.dimension == 8
    hits = loaded.search(vectors[5], k=1)
    assert hits[0].doc_id == "doc5"


def test_filtered_search(tmp_path: Path) -> None:
    vectors = make_vectors()
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(
        [f"doc{i}" for i in range(10)],
        vectors,
        [{"category": "a" if i < 5 else "b"} for i in range(10)],
    )
    hits = store.search(vectors[0], k=10, where={"category": "a"})
    assert hits
    assert all(hit.metadata["category"] == "a" for hit in hits)


def test_dimension_mismatch(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8, backend="tfidf")
    store.build(["a", "b"], make_vectors(2, 8))
    with pytest.raises(ValueError, match="backend"):
        store.search(np.zeros(4, dtype=np.float32), k=1)


def test_zero_norm_query_is_rejected(tmp_path: Path) -> None:
    """A query whose terms are all outside the vocabulary lands here."""
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(["a", "b"], make_vectors(2, 8))
    with pytest.raises(ValueError, match="zero vector"):
        store.search(np.zeros(8, dtype=np.float32), k=1)


def test_contains_and_get_vector(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8)
    vectors = make_vectors(2, 8)
    store.build(["a", "b"], vectors)

    assert store.contains("a")
    assert not store.contains("missing")
    assert np.allclose(store.get_vector("a"), store.vectors[0], atol=1e-6)
    with pytest.raises(KeyError):
        store.get_vector("missing")


def test_get_metadata(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(["a"], make_vectors(1, 8), [{"title": "A"}])

    assert store.get_metadata("a") == {"title": "A"}
    assert store.get_metadata("missing") is None


def test_add_rejects_wrong_metadata_length(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(["a"], make_vectors(1, 8))
    with pytest.raises(ValueError, match="Metadata length"):
        store.add(["b"], make_vectors(1, 8), [{"title": "B"}, {"title": "C"}])


def test_delete_removes_document(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8)
    store.build(["a", "b", "c"], make_vectors(3, 8))

    assert store.delete("b") is True
    assert store.delete("b") is False
    assert len(store) == 2
    assert store.doc_ids == ["a", "c"]
    assert store.vectors.shape == (2, 8)
    assert not store.contains("b")


def test_search_on_empty_index_raises(tmp_path: Path) -> None:
    store = VectorStore(path=tmp_path, dimension=8)
    with pytest.raises(RuntimeError, match="empty"):
        store.search(np.zeros(8, dtype=np.float32), k=1)
