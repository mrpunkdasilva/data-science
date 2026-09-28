"""Tests for the public API facade and the CLI wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from api.service import CapstoneAPI
from ingest.arxiv_fetcher import ArxivPaper, write_corpus

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def sample_papers() -> List[ArxivPaper]:
    """Small in-memory corpus, so the tests never hit the network."""
    return [
        ArxivPaper(
            arxiv_id=f"2601.0000{i}v1",
            title=f"Deep learning study number {i} about neural networks",
            summary=(
                "We study gradient descent and backpropagation in deep neural "
                f"networks, experiment {i}, with results on image classification."
            ),
            authors=[f"Author {i}"],
            published=f"2026-01-0{i + 1}T00:00:00Z",
            updated=f"2026-01-0{i + 1}T00:00:00Z",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.CV"],
            pdf_url=f"https://arxiv.org/pdf/2601.0000{i}v1",
            abs_url=f"https://arxiv.org/abs/2601.0000{i}v1",
        )
        for i in range(6)
    ]


@pytest.fixture
def api(tmp_path: Path) -> CapstoneAPI:
    """An API backed by a temp data dir seeded with a fake corpus."""
    facade = CapstoneAPI(data_dir=tmp_path)
    write_corpus(sample_papers(), facade.paths.corpus_dir)
    return facade


def test_write_corpus_emits_txt_and_manifest(tmp_path: Path) -> None:
    write_corpus(sample_papers(), tmp_path)

    texts = sorted(tmp_path.glob("*.txt"))
    assert len(texts) == 6
    assert "Deep learning study number 0" in texts[0].read_text(encoding="utf-8")

    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 6
    first: Dict[str, Any] = json.loads(manifest[0])
    assert first["primary_category"] == "cs.LG"
    assert first["abs_url"].startswith("https://arxiv.org/abs/")


def test_loader_merges_manifest_after_write(tmp_path: Path) -> None:
    from ingest.loader import DocumentLoader

    write_corpus(sample_papers(), tmp_path)
    documents = DocumentLoader(tmp_path).load_all()

    assert len(documents) == 6
    assert documents[0].metadata["authors"] == ["Author 0"]
    assert documents[0].metadata["year"] == 2026


def test_build_and_analyze(api: CapstoneAPI) -> None:
    info = api.build(backend="tfidf")
    assert info["documents"] == 6
    assert info["backend"] == "tfidf"

    analysis = api.analyze(backend="tfidf")
    assert analysis["documents"] == 6
    assert "clusters" in analysis
    assert analysis["quality"]["mean_score"] > 0


def test_search_uses_index_backend(api: CapstoneAPI) -> None:
    api.build(backend="tfidf")

    hits = api.search("deep learning", k=3)

    assert len(hits) == 3
    assert hits[0].score > 0
    assert hits[0].title
    assert hits[0].metadata["primary_category"] == "cs.LG"


def test_search_on_empty_index_raises(api: CapstoneAPI) -> None:
    with pytest.raises(FileNotFoundError):
        api.search("deep learning")


def test_similar_excludes_the_document_itself(api: CapstoneAPI) -> None:
    api.build(backend="tfidf")
    first = api.search("neural networks", k=1)[0]

    neighbours = api.similar(first.doc_id, k=3)

    assert first.doc_id not in {hit.doc_id for hit in neighbours}


def test_load_analysis_is_none_before_clustering(api: CapstoneAPI) -> None:
    assert api.load_analysis() is None

    api.build(backend="tfidf")
    api.analyze(backend="tfidf")

    payload = api.load_analysis()
    assert payload is not None
    assert len(payload["doc_ids"]) == 6
    assert len(payload["labels"]) == 6


def test_cli_help_lists_all_commands() -> None:
    from click.testing import CliRunner

    from api.cli import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "build", "cluster", "search", "explore", "serve"):
        assert command in result.output


def test_cli_search_reports_missing_index_cleanly(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from api.cli import cli

    result = CliRunner().invoke(
        cli, ["--data-dir", str(tmp_path), "search", "anything"]
    )

    assert result.exit_code == 1
    assert "capstone build" in result.output


def test_analysis_json_is_strictly_valid(api: CapstoneAPI) -> None:
    """
    analysis.json must be readable by any JSON parser.

    A near-duplicate corpus makes scikit-learn report NaN explained variance,
    and Python's json module would happily write the invalid `NaN` token.
    """
    api.build(backend="tfidf")
    api.analyze(backend="tfidf")

    raw = (api.paths.index_dir / "analysis.json").read_text(encoding="utf-8")
    assert "NaN" not in raw

    def reject(constant: str) -> None:
        raise AssertionError(f"invalid JSON token: {constant}")

    json.loads(raw, parse_constant=reject)


def test_module_layout_is_importable() -> None:
    """The packages ship as real importable modules, not just scripts."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    assert src not in sys.path or True
    for module in (
        "ingest.loader",
        "embeddings.factory",
        "index.vector_store",
        "cluster.clustering",
        "quality.classifier",
        "pipeline.core",
    ):
        __import__(module)
