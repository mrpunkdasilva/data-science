# -*- coding: utf-8 -*-
"""Document loader: reads .txt, .md, and .pdf files from a directory."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pdf_reader import extract_pdf_metadata, read_pdf
from .text_cleaner import clean_text

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MANIFEST_NAME = "manifest.jsonl"

# Manifest keys that are already represented by dedicated Document fields and
# are therefore not copied into the free-form metadata.
_RESERVED_KEYS = {"doc_id", "title"}


@dataclass
class Document:
    """A loaded document with metadata."""

    doc_id: str
    title: str
    text: str
    source_path: str
    source_type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentLoader:
    """
    Load documents from a directory or a single file.

    When a ``manifest.jsonl`` sits next to the documents, its records are
    merged into each document's metadata. That is how the arXiv ingest keeps
    authors, categories and links: the ``.txt`` file only holds title and
    abstract, which is what gets embedded, while the manifest holds the
    structured fields used for filtering and display.
    """

    def __init__(self, corpus_dir: Path, manifest_name: str = MANIFEST_NAME):
        self.corpus_dir = Path(corpus_dir)
        self.manifest_name = manifest_name
        self._manifest: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def manifest(self) -> Dict[str, Dict[str, Any]]:
        """Manifest records keyed by doc_id, read once and cached."""
        if self._manifest is None:
            self._manifest = self._read_manifest()
        return self._manifest

    def _read_manifest(self) -> Dict[str, Dict[str, Any]]:
        """Parse manifest.jsonl, skipping malformed lines."""
        path = self.corpus_dir / self.manifest_name
        records: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return records

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = record.get("doc_id")
                if isinstance(doc_id, str) and doc_id:
                    records[doc_id] = record
        return records

    def load_all(self) -> List[Document]:
        """Load all supported documents from corpus_dir."""
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        documents: List[Document] = []
        for path in sorted(self.corpus_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc = self.load_one(path)
                if doc is not None and doc.text:
                    documents.append(doc)
        return documents

    def load_one(self, path: Path) -> Optional[Document]:
        """Load a single document, enriched with its manifest record."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return None

        metadata: Dict[str, Any] = {}
        if ext == ".pdf":
            try:
                raw_text = read_pdf(path)
            except RuntimeError:
                return None
            doc_id = path.stem
            source_type = "pdf"
            title = path.stem
            metadata = extract_pdf_metadata(path)
        else:
            try:
                raw_text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return None
            doc_id = path.stem
            source_type = "text"
            title = _extract_title_from_text(raw_text) or path.stem

        manifest_record = self.manifest.get(doc_id)
        if manifest_record:
            title = str(manifest_record.get("title") or title)
            metadata = {
                **{
                    key: value
                    for key, value in manifest_record.items()
                    if key not in _RESERVED_KEYS
                },
                **metadata,
            }
            year = _year_from(manifest_record.get("published"))
            if year is not None:
                metadata.setdefault("year", year)

        return Document(
            doc_id=doc_id,
            title=title,
            text=clean_text(raw_text),
            source_path=str(path),
            source_type=source_type,
            metadata=metadata,
        )


def _year_from(published: Any) -> Optional[int]:
    """Pull the year out of an ISO-8601 timestamp, when present."""
    if not isinstance(published, str) or len(published) < 4:
        return None
    try:
        return int(published[:4])
    except ValueError:
        return None


def _extract_title_from_text(text: str, max_lines: int = 5) -> Optional[str]:
    """Try to extract a title from the first few lines of a text file."""
    for line in text.split("\n")[:max_lines]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:200]
        if stripped.startswith("# "):
            return stripped[2:].strip()[:200]
    return None
