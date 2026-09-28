"""PDF reader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pypdf


def read_pdf(path: Path) -> str:
    """Extract text from a PDF file, skipping pages that fail to parse."""
    text_parts: List[str] = []
    try:
        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
            if extracted:
                text_parts.append(extracted)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF {path}: {exc}") from exc

    return "\n\n".join(text_parts)


def extract_pdf_metadata(path: Path) -> Dict[str, Any]:
    """Extract basic PDF metadata, returning an empty dict when unavailable."""
    try:
        reader = pypdf.PdfReader(str(path))
        info = reader.metadata
        if not info:
            return {}
        return {
            "title": str(info.get("/Title", "") or ""),
            "author": str(info.get("/Author", "") or ""),
            "creator": str(info.get("/Creator", "") or ""),
            "producer": str(info.get("/Producer", "") or ""),
        }
    except Exception:
        return {}
