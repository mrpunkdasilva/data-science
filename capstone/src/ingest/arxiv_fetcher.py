# -*- coding: utf-8 -*-
"""ArXiv API client for fetching scientific paper metadata."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}
API_URL = "https://export.arxiv.org/api/query"
MANIFEST_FILE = "manifest.jsonl"


@dataclass
class ArxivPaper:
    """A single arXiv paper."""

    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    updated: str
    primary_category: str
    categories: List[str]
    pdf_url: str
    abs_url: str

    @property
    def text_for_embedding(self) -> str:
        """Combined text used for embedding."""
        return f"{self.title}\n\n{self.summary}"


def fetch_papers(
    category: str = "cs.LG",
    max_results: int = 500,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    start: int = 0,
    delay: float = 3.0,
) -> List[ArxivPaper]:
    """
    Fetch paper metadata from the arXiv API.

    Respects arXiv's rate limit of 1 request per 3 seconds.
    """
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{API_URL}?{urlencode(params)}"

    request = Request(url, headers={"User-Agent": "csds-352-capstone/1.0"})
    with urlopen(request, timeout=30) as response:
        xml_content = response.read()

    root = ET.fromstring(xml_content)
    entries = root.findall("atom:entry", ATOM_NS)

    papers: List[ArxivPaper] = []
    for entry in entries:
        raw_id = _find_text(entry, "atom:id", ATOM_NS)
        if not raw_id:
            continue

        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id

        authors = [
            a.text for a in entry.findall("atom:author/atom:name", ATOM_NS) if a.text
        ]

        primary = entry.find("arxiv:primary_category", ARXIV_NS)
        primary_cat = primary.get("term", "") if primary is not None else ""

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", ATOM_NS)
            if c.get("term")
        ]

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break

        abs_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("rel") == "alternate":
                abs_url = link.get("href", "")
                break

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=_clean_inline(_find_text(entry, "atom:title", ATOM_NS) or ""),
                summary=(_find_text(entry, "atom:summary", ATOM_NS) or "").strip(),
                authors=authors,
                published=_find_text(entry, "atom:published", ATOM_NS) or "",
                updated=_find_text(entry, "atom:updated", ATOM_NS) or "",
                primary_category=primary_cat,
                categories=categories,
                pdf_url=pdf_url,
                abs_url=abs_url,
            )
        )

    return papers


def _find_text(entry: ET.Element, path: str, ns: Dict[str, str]) -> Optional[str]:
    """Helper to find text in an XML element."""
    el = entry.find(path, ns)
    return el.text if el is not None and el.text else None


def _clean_inline(text: str) -> str:
    """Remove newlines from inline text like titles."""
    return " ".join(text.split())


def write_corpus(papers: List[ArxivPaper], target: Path) -> List[Dict[str, Any]]:
    """
    Persist fetched papers as one .txt per paper plus a manifest.

    The .txt holds only title and abstract, because that is what gets embedded:
    an embedding of a citation list or a PDF header is mostly noise. Everything
    structured (authors, categories, links) goes to manifest.jsonl, which the
    loader merges back as metadata at read time.

    Returns the manifest records that were written.
    """
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / MANIFEST_FILE
    records: List[Dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for paper in papers:
            doc_id = paper.arxiv_id.replace("/", "_")
            (target / f"{doc_id}.txt").write_text(
                paper.text_for_embedding, encoding="utf-8"
            )
            record: Dict[str, Any] = {
                "doc_id": doc_id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "abs_url": paper.abs_url,
                "source": "arxiv",
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    return records
