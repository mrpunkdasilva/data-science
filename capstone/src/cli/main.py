"""Main CLI entry point for the capstone pipeline."""

import click
from pathlib import Path
from rich.console import Console

from ingest.loader import load_reviews
from embed.model import embed_reviews
from store.client import get_chroma_client, get_collection
from cluster.pipeline import run_clustering
from quality.classifier import run_quality_classification
from quality.anomaly import run_anomaly_detection
from search.engine import search_reviews, list_clusters, get_anomalies, get_stats

console = Console()


@click.group()
def cli() -> None:
    """Amazon Reviews Analysis Pipeline - Capstone Project"""
    pass


@cli.command()
@click.option(
    "--categories",
    "-c",
    default="raw_review_Electronics,raw_review_Home_and_Kitchen",
    help="Comma-separated HF dataset categories",
)
@click.option("--limit", "-l", default=15000, help="Total reviews to load")
@click.option(
    "--output", "-o", default="data/processed/reviews.parquet", help="Output parquet path"
)
@click.option("--min-len", default=50, help="Minimum review text length")
@click.option("--max-len", default=500, help="Maximum review text length")
@click.option("--verified-only/--no-verified-only", default=True, help="Only verified purchases")
def ingest(
    categories: str,
    limit: int,
    output: str,
    min_len: int,
    max_len: int,
    verified_only: bool,
) -> None:
    """Load reviews from HuggingFace datasets and save to parquet."""
    cat_list = [c.strip() for c in categories.split(",")]
    console.print(
        f"[bold green]Loading {limit} reviews from {len(cat_list)} categories...[/bold green]"
    )
    df = load_reviews(
        categories=cat_list,
        limit=limit,
        min_text_len=min_len,
        max_text_len=max_len,
        verified_only=verified_only,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    console.print(f"[bold green]Saved {len(df)} reviews to {output}[/bold green]")


@cli.command()
@click.option("--input", "-i", default="data/processed/reviews.parquet", help="Input parquet path")
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--batch-size", "-b", default=32, help="Embedding batch size")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
def embed(
    input: str,
    collection: str,
    batch_size: int,
    device: str,
) -> None:
    """Generate embeddings and store in ChromaDB."""
    console.print(f"[bold green]Embedding reviews from {input} into {collection}...[/bold green]")
    embed_reviews(
        input_path=input,
        collection_name=collection,
        batch_size=batch_size,
        device=device,
    )
    console.print("[bold green]Embedding complete![/bold green]")


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--umap-neighbors", default=15, help="UMAP n_neighbors")
@click.option("--umap-min-dist", default=0.1, help="UMAP min_dist")
@click.option("--hdbscan-min-cluster", default=15, help="HDBSCAN min_cluster_size")
@click.option("--hdbscan-min-samples", default=5, help="HDBSCAN min_samples")
def cluster(
    collection: str,
    umap_neighbors: int,
    umap_min_dist: float,
    hdbscan_min_cluster: int,
    hdbscan_min_samples: int,
) -> None:
    """Run UMAP + HDBSCAN clustering on stored embeddings."""
    console.print(f"[bold green]Clustering collection {collection}...[/bold green]")
    run_clustering(
        collection_name=collection,
        umap_n_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        hdbscan_min_cluster_size=hdbscan_min_cluster,
        hdbscan_min_samples=hdbscan_min_samples,
    )
    console.print("[bold green]Clustering complete![/bold green]")


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--contamination", default=0.05, help="Anomaly contamination ratio")
def quality(
    collection: str,
    contamination: float,
) -> None:
    """Run quality classification and anomaly detection."""
    console.print(f"[bold green]Running quality control on {collection}...[/bold green]")
    run_quality_classification(collection_name=collection)
    run_anomaly_detection(collection_name=collection, contamination=contamination)
    console.print("[bold green]Quality control complete![/bold green]")


@cli.command()
@click.argument("query")
@click.option("--k", "-k", default=10, help="Number of results")
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--category", default="", help="Filter by category")
@click.option("--rating-min", default=0, help="Minimum rating (1-5)")
@click.option("--quality", default="", help="Filter by quality label")
def search(
    query: str,
    k: int,
    collection: str,
    category: str,
    rating_min: int,
    quality: str,
) -> None:
    """Semantic search for similar reviews."""
    cat = category if category else None
    rating = rating_min if rating_min > 0 else None
    qual = quality if quality else None

    results = search_reviews(
        query=query,
        k=k,
        collection_name=collection,
        category=cat,
        rating_min=rating,
        quality_label=qual,
    )
    for i, r in enumerate(results, 1):
        console.print(
            f"\n[bold cyan]{i}.[/bold cyan] [yellow]{r['title']}[/yellow] (score: {r['score']:.3f})"
        )
        console.print(
            f"  Category: {r['category']} | Rating: {r['rating']} | Helpful: {r['helpful_vote']}"
        )
        console.print(
            f"  Quality: {r['quality_label']} | Cluster: {r['cluster_id']} | Anomaly: {r['anomaly_score']:.3f}"
        )
        console.print(f"  {r['text'][:200]}...")


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--top-terms", default=5, help="Top terms per cluster")
def clusters(
    collection: str,
    top_terms: int,
) -> None:
    """List all clusters with statistics."""
    cluster_info = list_clusters(collection_name=collection, top_terms=top_terms)
    for c in cluster_info:
        console.print(
            f"\n[bold magenta]Cluster {c['cluster_id']}[/bold magenta] ({c['size']} reviews)"
        )
        console.print(f"  Avg Rating: {c['avg_rating']:.2f} | Avg Helpful: {c['avg_helpful']:.1f}")
        console.print(f"  Dominant Quality: {c['dominant_quality']}")
        console.print(f"  Top Terms: {', '.join(c['top_terms'])}")


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--limit", "-l", default=20, help="Number of anomalies to show")
@click.option("--min-score", default=0.0, help="Minimum anomaly score")
def anomalies(
    collection: str,
    limit: int,
    min_score: float,
) -> None:
    """Show top anomalous reviews."""
    results = get_anomalies(collection_name=collection, limit=limit, min_score=min_score)
    for i, r in enumerate(results, 1):
        console.print(f"\n[bold red]{i}.[/bold red] Anomaly Score: {r['anomaly_score']:.3f}")
        console.print(f"  Title: {r['title']}")
        console.print(
            f"  Category: {r['category']} | Rating: {r['rating']} | Quality: {r['quality_label']}"
        )
        console.print(f"  {r['text'][:300]}...")


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
def stats(
    collection: str,
) -> None:
    """Show collection statistics."""
    stats = get_stats(collection_name=collection)
    console.print(f"\n[bold]Collection:[/bold] {stats['collection']}")
    console.print(f"[bold]Total Reviews:[/bold] {stats['total_reviews']}")
    console.print(f"[bold]Categories:[/bold] {stats['categories']}")
    console.print(f"[bold]Clusters:[/bold] {stats['n_clusters']} (noise: {stats['noise_count']})")
    console.print(f"[bold]Quality Distribution:[/bold]")
    for label, count in stats["quality_distribution"].items():
        console.print(f"  {label}: {count}")
    console.print(f"[bold]Anomalies:[/bold] {stats['anomaly_count']} ({stats['anomaly_pct']:.1f}%)")


@cli.command()
@click.option("--limit", "-l", default=15000, help="Total reviews to process")
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--skip-ingest", is_flag=True, help="Skip ingestion step")
@click.option("--skip-embed", is_flag=True, help="Skip embedding step")
@click.option("--skip-cluster", is_flag=True, help="Skip clustering step")
@click.option("--skip-quality", is_flag=True, help="Skip quality step")
def pipeline(
    limit: int,
    collection: str,
    skip_ingest: bool,
    skip_embed: bool,
    skip_cluster: bool,
    skip_quality: bool,
) -> None:
    """Run full pipeline: ingest -> embed -> cluster -> quality."""
    if not skip_ingest:
        console.print("[bold blue]=== INGEST ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            ingest,
            categories="raw_review_Electronics,raw_review_Home_and_Kitchen",
            limit=limit,
            output="data/processed/reviews.parquet",
            min_len=50,
            max_len=500,
            verified_only=True,
        )
    if not skip_embed:
        console.print("[bold blue]=== EMBED ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            embed,
            input="data/processed/reviews.parquet",
            collection=collection,
            batch_size=32,
            device="cpu",
        )
    if not skip_cluster:
        console.print("[bold blue]=== CLUSTER ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            cluster,
            collection=collection,
            umap_neighbors=15,
            umap_min_dist=0.1,
            hdbscan_min_cluster=15,
            hdbscan_min_samples=5,
        )
    if not skip_quality:
        console.print("[bold blue]=== QUALITY ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(quality, collection=collection, contamination=0.05)
    console.print("[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    cli()
