"""Command line interface for the capstone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from embeddings.factory import BACKENDS
from index.search import SemanticSearch
from index.vector_store import VectorStore
from ingest.arxiv_fetcher import fetch_papers, write_corpus
from ingest.text_cleaner import slugify
from pipeline import Pipeline, PipelinePaths
from lib.console_manager import console_manager as cm

DEFAULT_DATA_DIR = Path("capstone/data")


def _paths(data_dir: Path) -> PipelinePaths:
    return PipelinePaths.under(data_dir)


def _backend_from_manifest(paths: PipelinePaths) -> str:
    """Read which embedding backend built the index, so search matches it."""
    manifest = paths.index_dir / "manifest.json"
    if not manifest.exists():
        return "tfidf"
    try:
        with manifest.open(encoding="utf-8") as handle:
            return str(json.load(handle).get("backend", "tfidf"))
    except (OSError, ValueError):
        return "tfidf"


@click.group()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    help="Directory holding corpus/, index/ and models/.",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: Path) -> None:
    """Explore a corpus of scientific papers with embeddings and clustering."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@cli.command()
@click.option("--category", default="cs.LG", help="arXiv category, e.g. cs.LG, cs.CL.")
@click.option("--max-results", default=500, type=int, help="How many papers to fetch.")
@click.option("--sort-by", default="submittedDate", help="arXiv sort field.")
@click.option("--sort-order", default="descending", help="ascending or descending.")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save the .txt corpus (defaults to <data-dir>/corpus).",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    category: str,
    max_results: int,
    sort_by: str,
    sort_order: str,
    out_dir: Optional[Path],
) -> None:
    """Download paper metadata from arXiv and save one .txt per paper."""
    paths = _paths(ctx.obj["data_dir"])
    target = Path(out_dir) if out_dir else paths.corpus_dir
    target.mkdir(parents=True, exist_ok=True)

    cm.print_info(f"Fetching up to {max_results} papers from arXiv ({category})...")
    papers = fetch_papers(
        category=category,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    records = write_corpus(papers, target)
    cm.print_success(f"Ingested {len(records)} papers into {target}")


@cli.command()
@click.option(
    "--backend", type=click.Choice(BACKENDS), default="tfidf", help="Embedding backend."
)
@click.option(
    "--model-name", default=None, help="Override the sentence-transformers model."
)
@click.option("--batch-size", default=32, type=int, help="Encoding batch size.")
@click.pass_context
def build(
    ctx: click.Context,
    backend: str,
    model_name: Optional[str],
    batch_size: int,
) -> None:
    """Embed the corpus and persist the vector index."""
    paths = _paths(ctx.obj["data_dir"])
    pipeline = Pipeline(
        paths, backend=backend, model_name=model_name, batch_size=batch_size
    )

    documents = pipeline.load_documents()
    if not documents:
        cm.print_error("No documents found. Run 'capstone ingest' first.")
        raise SystemExit(1)
    cm.print_info(f"Loaded {len(documents)} documents. Embedding with '{backend}'...")

    _store, info = pipeline.build_index(documents)
    cm.print_success(
        f"Index built: {info['documents']} vectors, dim {info['dimension']} ({info['backend']})"
    )


@cli.command()
@click.option(
    "--algorithm",
    default="kmeans",
    type=click.Choice(["kmeans", "hdbscan", "agglomerative"]),
    help="Clustering algorithm.",
)
@click.option("--n-clusters", default=8, type=int, help="Number of clusters (kmeans).")
@click.option(
    "--projection",
    default="pca",
    type=click.Choice(["pca", "umap", "none"]),
    help="Projection used for the 2D map.",
)
@click.option(
    "--contamination", default=0.05, type=float, help="Expected share of anomalies."
)
@click.pass_context
def cluster(
    ctx: click.Context,
    algorithm: str,
    n_clusters: int,
    projection: str,
    contamination: float,
) -> None:
    """Cluster the index, project to 2D and score document quality."""
    paths = _paths(ctx.obj["data_dir"])
    if not (paths.index_dir / "embeddings.npy").exists():
        cm.print_error("No index found. Run 'capstone build' first.")
        raise SystemExit(1)

    store = VectorStore.load(paths.index_dir)
    pipeline = Pipeline(paths, backend=_backend_from_manifest(paths))
    cm.print_info(f"Clustering {len(store)} documents with {algorithm}...")
    analysis = pipeline.analyze(
        store,
        algorithm=algorithm,
        n_clusters=n_clusters,
        projection=projection,
        contamination=contamination,
    )

    cm.print_success(f"Clusters: {analysis['clusters']}")
    cm.print_success(f"Quality: {analysis['quality']['distribution']}")
    cm.print_success(
        f"Anomalies: {analysis['anomalies']['anomalies']} of {analysis['anomalies']['total']}"
    )


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--k", default=5, type=int, help="Number of results.")
@click.option(
    "--backend",
    type=click.Choice(BACKENDS),
    default=None,
    help="Embedding backend (defaults to the one used to build the index).",
)
@click.pass_context
def search(
    ctx: click.Context, query: List[str], k: int, backend: Optional[str]
) -> None:
    """Semantic search over the indexed corpus."""
    paths = _paths(ctx.obj["data_dir"])
    try:
        store = VectorStore.load(paths.index_dir)
    except FileNotFoundError as exc:
        cm.print_error(str(exc))
        raise SystemExit(1) from exc

    chosen = backend or store.backend
    searcher = SemanticSearch(store, backend=chosen)
    text = " ".join(query)
    cm.print_info(f'Searching for: "{text}" (backend: {chosen})')
    try:
        hits = searcher.query(text, k=k)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        cm.print_error(str(exc))
        raise SystemExit(1) from exc

    if not hits:
        cm.print_info("No document matched this query.")
        return

    for rank, hit in enumerate(hits, start=1):
        cm.print_markdown(
            f"**{rank}. {hit.title}**  \n"
            f"similarity `{hit.score:.3f}` · `{hit.doc_id}`"
        )


@cli.command()
@click.option("--top", default=20, type=int, help="How many clusters to show.")
@click.pass_context
def explore(ctx: click.Context, top: int) -> None:
    """Show a summary of the analysed index."""
    paths = _paths(ctx.obj["data_dir"])
    analysis_path = paths.index_dir / "analysis.json"
    if not analysis_path.exists():
        cm.print_error("No analysis found. Run 'capstone cluster' first.")
        raise SystemExit(1)

    with analysis_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    analysis = payload["analysis"]
    cm.print_dict(
        {
            "documents": analysis["documents"],
            "dimension": analysis["dimension"],
            "clusters": len(analysis["clusters"]),
            "mean_quality": f"{analysis['quality']['mean_score']:.3f}",
            "anomalies": analysis["anomalies"]["anomalies"],
        },
        header="Corpus overview",
    )

    for label, size in list(analysis["clusters"].items())[:top]:
        cm.print_info(f"cluster {label}: {size} documents")


@cli.command()
@click.option("--port", default=8501, type=int, help="Streamlit port.")
def serve(port: int) -> None:
    """Launch the Streamlit exploration UI."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py"
    if not app_path.exists():
        cm.print_error(f"UI script not found at {app_path}")
        raise SystemExit(1)

    cm.print_info(f"Starting Streamlit UI on port {port}...")
    # Headless mode skips the first-run email prompt, which otherwise blocks the
    # server for anyone running the project on a machine that never used
    # Streamlit, such as a grader's laptop or CI.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        check=False,
    )


if __name__ == "__main__":
    cli()
