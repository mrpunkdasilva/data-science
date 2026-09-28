"""Document ingestion: arXiv fetching, PDF reading and text cleaning."""

from .arxiv_fetcher import ArxivPaper, fetch_papers
from .loader import Document, DocumentLoader
from .pdf_reader import read_pdf
from .text_cleaner import clean_text, split_into_chunks, slugify

__all__ = [
    "ArxivPaper",
    "fetch_papers",
    "Document",
    "DocumentLoader",
    "read_pdf",
    "clean_text",
    "split_into_chunks",
    "slugify",
]
