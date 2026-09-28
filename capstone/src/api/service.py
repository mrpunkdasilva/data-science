"""HTTP-style API facade over the pipeline, usable from notebooks or services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from index.search import SemanticSearch
from index.vector_store import SearchHit, VectorStore
from pipeline import Pipeline, PipelinePaths


class CapstoneAPI:
    """
    Programmatic entry point, the API counterpart of the CLI.

    The CLI and this class share the same pipeline, so anything you can do
    from the terminal you can also do from Python or a notebook.
    """

    def __init__(self, data_dir: Path | str = "capstone/data") -> None:
        self.paths = PipelinePaths.under(Path(data_dir))

    def ingest(
        self, category: str = "cs.LG", max_results: int = 500, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv and persist them locally."""
        from ingest.arxiv_fetcher import fetch_papers, write_corpus

        papers = fetch_papers(category=category, max_results=max_results, **kwargs)
        return write_corpus(papers, self.paths.corpus_dir)

    def build(self, backend: str = "tfidf", **kwargs: Any) -> Dict[str, Any]:
        """Embed the corpus and persist the index."""
        pipeline = Pipeline(self.paths, backend=backend, **kwargs)
        documents = pipeline.load_documents()
        if not documents:
            raise RuntimeError("Corpus is empty, call ingest() first")
        _store, info = pipeline.build_index(documents)
        return info

    def analyze(self, backend: str = "tfidf") -> Dict[str, Any]:
        """Cluster, project and score an existing index."""
        store = VectorStore.load(self.paths.index_dir)
        pipeline = Pipeline(self.paths, backend=backend)
        return pipeline.analyze(store)

    def search(
        self, query: str, k: int = 5, backend: Optional[str] = None
    ) -> List[SearchHit]:
        """Semantic search from Python, using the backend the index was built with."""
        store = VectorStore.load(self.paths.index_dir)
        return SemanticSearch(store, backend=backend or store.backend).query(query, k=k)

    def similar(
        self, doc_id: str, k: int = 5, backend: Optional[str] = None
    ) -> List[SearchHit]:
        """More-like-this search from Python."""
        store = VectorStore.load(self.paths.index_dir)
        return SemanticSearch(store, backend=backend or store.backend).similar(
            doc_id, k=k
        )

    def load_analysis(self) -> Optional[Dict[str, Any]]:
        """Read the cluster/quality analysis produced by analyze()."""
        import json

        path = self.paths.index_dir / "analysis.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            loaded: Dict[str, Any] = json.load(handle)
            return loaded
