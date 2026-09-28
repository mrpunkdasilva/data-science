"""Tests for document loading and the arXiv client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.loader import DocumentLoader
from ingest.pdf_reader import read_pdf


def test_load_text_documents(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Title A\n\nBody of document A.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title B\n\nBody of document B.", encoding="utf-8")

    documents = DocumentLoader(tmp_path).load_all()
    assert len(documents) == 2
    assert {doc.doc_id for doc in documents} == {"a", "b"}


def test_loader_ignores_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    documents = DocumentLoader(tmp_path).load_all()
    assert len(documents) == 1


def test_loader_skips_empty(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "full.txt").write_text("real content", encoding="utf-8")

    documents = DocumentLoader(tmp_path).load_all()
    assert len(documents) == 1


def test_load_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DocumentLoader(tmp_path / "nope").load_all()


def test_read_pdf_invalid_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(RuntimeError):
        read_pdf(broken)


def test_manifest_metadata_is_merged(tmp_path: Path) -> None:
    (tmp_path / "2601.00001v1.txt").write_text(
        "Short text title\n\nAn abstract.", encoding="utf-8"
    )
    record = {
        "doc_id": "2601.00001v1",
        "title": "The Real Title From The Manifest",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "published": "2026-01-02T10:00:00Z",
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "stat.ML"],
        "abs_url": "https://arxiv.org/abs/2601.00001v1",
    }
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    document = DocumentLoader(tmp_path).load_one(tmp_path / "2601.00001v1.txt")

    assert document is not None
    assert document.title == "The Real Title From The Manifest"
    assert document.metadata["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert document.metadata["primary_category"] == "cs.LG"
    assert document.metadata["year"] == 2026


def test_manifest_is_optional(tmp_path: Path) -> None:
    (tmp_path / "plain.txt").write_text("Plain title\n\nBody.", encoding="utf-8")

    document = DocumentLoader(tmp_path).load_one(tmp_path / "plain.txt")

    assert document is not None
    assert document.title == "Plain title"
    assert document.metadata == {}


def test_manifest_skips_malformed_lines(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Title A\n\nBody.", encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text(
        "not json\n" + json.dumps({"doc_id": "a", "title": "From manifest"}) + "\n",
        encoding="utf-8",
    )

    document = DocumentLoader(tmp_path).load_one(tmp_path / "a.txt")

    assert document is not None
    assert document.title == "From manifest"
