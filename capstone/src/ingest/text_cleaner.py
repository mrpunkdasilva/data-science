# -*- coding: utf-8 -*-
"""Utilities for text cleaning and normalization."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize raw text."""
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str, max_chars: int = 2000, overlap: int = 200
) -> List[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at sentence end
        if end < len(text):
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                idx = text.rfind(sep, start, end)
                if idx > start + max_chars // 2:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def slugify(s: str) -> str:
    """Create a filesystem-friendly slug."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "document"
