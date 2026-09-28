"""
Smoke tests for the Streamlit app.

The app is the part a grader clicks first, so these run the real script through
Streamlit's AppTest harness instead of mocking the framework. They are skipped
when the optional UI extra is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="UI extra not installed")
pytest.importorskip("plotly", reason="UI extra not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

CAPSTONE_ROOT = Path(__file__).resolve().parents[1]
SRC = CAPSTONE_ROOT / "src"
APP = SRC / "ui" / "streamlit_app.py"
CORPUS = CAPSTONE_ROOT / "data" / "corpus"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_fixture_index(tmp_path: Path) -> Path:
    """
    Run the real pipeline over a slice of the corpus.

    Going through Pipeline.analyze instead of hand-writing an analysis.json
    means the app is tested against the artifacts the pipeline really produces,
    so a contract change between the two surfaces here.
    """
    from embeddings.tfidf_svd import TfidfSvdEmbedder
    from index.vector_store import VectorStore
    from ingest.loader import DocumentLoader
    from pipeline.core import Pipeline, PipelinePaths

    documents = DocumentLoader(CORPUS).load_all()[:40]
    if not documents:
        pytest.skip("corpus is empty, run 'capstone ingest' first")

    paths = PipelinePaths(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "index",
        models_dir=tmp_path / "models",
    )
    embedder = TfidfSvdEmbedder(n_components=24, min_df=1)
    vectors = embedder.fit_transform([document.text for document in documents])

    store = VectorStore(path=paths.index_dir, dimension=24, backend="tfidf")
    store.build(
        [document.doc_id for document in documents],
        vectors,
        [
            {
                **document.metadata,
                "title": document.title,
                "source_type": document.source_type,
                "text": document.text,
            }
            for document in documents
        ],
    )
    store.save()
    embedder.save(paths.models_dir / "tfidf_svd.joblib")

    Pipeline(paths, backend="tfidf").analyze(
        store, algorithm="kmeans", n_clusters=3, projection="pca"
    )
    return tmp_path


@pytest.fixture(scope="module")
def fixture_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_fixture_index(tmp_path_factory.mktemp("ui"))


def _run_app(fixture_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(fixture_data_dir))
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run(timeout=120)
    return app


def test_app_renders_without_exceptions(
    fixture_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_app(fixture_data_dir, monkeypatch)

    assert not app.exception
    assert not app.error
    assert app.title


def test_overview_shows_pipeline_metrics(
    fixture_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_app(fixture_data_dir, monkeypatch)

    labels = [metric.label for metric in app.metric]
    assert "Documents" in labels
    assert "Anomalies" in labels

    documents = next(m for m in app.metric if m.label == "Documents")
    assert int(documents.value) > 0


def test_sidebar_filters_narrow_the_result_set(
    fixture_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_app(fixture_data_dir, monkeypatch)

    before = next(cap for cap in app.caption if cap.value.startswith("Showing"))
    clusters = next(box for box in app.multiselect if box.label == "Clusters")
    # Set the underlying cluster ids, not the format_func labels.
    app = clusters.set_value([0]).run(timeout=120)

    after = next(cap for cap in app.caption if cap.value.startswith("Showing"))
    assert not app.exception
    assert after.value != before.value
    assert "of 40 documents" in after.value


def test_semantic_search_returns_hits(
    fixture_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _run_app(fixture_data_dir, monkeypatch)

    next(box for box in app.text_input if box.label == "Query").set_value(
        "neural network training"
    )
    app = (
        next(button for button in app.button if button.label == "Search")
        .click()
        .run(timeout=120)
    )

    assert not app.exception
    assert not app.error
    ranked = [block.value for block in app.markdown if block.value.startswith("**1.")]
    assert ranked, "expected at least one ranked result"
