"""Main CLI entry point for the capstone pipeline."""

import click
from pathlib import Path
from rich.console import Console

from ingest.tst import load_verbetes
from embed.model import embed_reviews
from store.client import get_chroma_client, get_collection
from cluster.pipeline import run_clustering
from quality.classifier import run_quality_classification
from quality.anomaly import run_anomaly_detection
from search.engine import search_reviews, list_clusters, get_anomalies, get_stats
from export.jsonl import export_jsonl

console = Console()


@click.group()
def cli() -> None:
    """TST Livro de Jurisprudência Analysis Pipeline - Capstone Project"""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    default="data/raw/livrointernet12pdf.pdf",
    help="Path to the TST Livro de Jurisprudência PDF",
)
@click.option(
    "--output",
    "-o",
    default="data/processed/tst.parquet",
    help="Output parquet path",
)
@click.option("--min-len", default=20, help="Minimum verbete text length")
def ingest(
    input: str,
    output: str,
    min_len: int,
) -> None:
    """Load verbetes (Súmulas, OJs, PNs) from the TST PDF and save to parquet."""
    console.print(f"[bold green]Parsing verbetes from {input}...[/bold green]")
    df = load_verbetes(pdf_path=input, min_text_len=min_len)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    console.print(f"[bold green]Saved {len(df)} verbetes to {output}[/bold green]")


@cli.command()
@click.option(
    "--input", "-i", default="data/processed/tst.parquet", help="Input parquet path"
)
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--batch-size", "-b", default=32, help="Embedding batch size")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
@click.option("--id-prefix", default="verbete", help="Record id prefix")
def embed(
    input: str,
    collection: str,
    batch_size: int,
    device: str,
    id_prefix: str,
) -> None:
    """Generate embeddings and store in ChromaDB."""
    console.print(
        f"[bold green]Embedding verbetes from {input} into {collection}...[/bold green]"
    )
    embed_reviews(
        input_path=input,
        collection_name=collection,
        batch_size=batch_size,
        device=device,
        id_prefix=id_prefix,
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
    console.print(
        f"[bold green]Running quality control on {collection}...[/bold green]"
    )
    run_quality_classification(collection_name=collection)
    run_anomaly_detection(collection_name=collection, contamination=contamination)
    console.print("[bold green]Quality control complete![/bold green]")


@cli.command()
@click.argument("query")
@click.option("--k", "-k", default=10, help="Number of results")
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--tipo", default="", help="Filter by tipo (sumula/oj/precedente)")
@click.option(
    "--orgao", default="", help="Filter by orgão (TST/SBDI-1/SBDI-2/SDC/TP-OE)"
)
@click.option("--quality", default="", help="Filter by quality label")
def search(
    query: str,
    k: int,
    collection: str,
    tipo: str,
    orgao: str,
    quality: str,
) -> None:
    """Semantic search for similar verbetes."""
    t = tipo if tipo else None
    org = orgao if orgao else None
    qual = quality if quality else None

    results = search_reviews(
        query=query,
        k=k,
        collection_name=collection,
        tipo=t,
        orgao=org,
        quality_label=qual,
    )
    for i, r in enumerate(results, 1):
        console.print(
            f"\n[bold cyan]{i}.[/bold cyan] [yellow]{r['tema']}[/yellow] (score: {r['score']:.3f})"
        )
        console.print(
            f"  Tipo: {r['tipo']} | Orgão: {r['orgao']} | Nº: {r['codigo']} | {r['doc_id']}"
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
            f"\n[bold magenta]Cluster {c['cluster_id']}[/bold magenta] ({c['size']} verbetes)"
        )
        console.print(
            f"  Dominant Quality: {c['dominant_quality']} | Tipos: {c['tipos']}"
        )
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
    """Show top anomalous verbetes."""
    results = get_anomalies(
        collection_name=collection, limit=limit, min_score=min_score
    )
    for i, r in enumerate(results, 1):
        console.print(
            f"\n[bold red]{i}.[/bold red] Anomaly Score: {r['anomaly_score']:.3f}"
        )
        console.print(f"  Tema: {r['tema']}")
        console.print(
            f"  Tipo: {r['tipo']} | Orgão: {r['orgao']} | Nº: {r['codigo']} | Quality: {r['quality_label']}"
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
    console.print(f"[bold]Total Documents:[/bold] {stats['total_documents']}")
    console.print(f"[bold]By Tipo:[/bold] {stats['tipos']}")
    console.print(f"[bold]By Orgão:[/bold] {stats['orgaos']}")
    console.print(
        f"[bold]Clusters:[/bold] {stats['n_clusters']} (noise: {stats['noise_count']})"
    )
    console.print(f"[bold]Quality Distribution:[/bold]")
    for label, count in stats["quality_distribution"].items():
        console.print(f"  {label}: {count}")
    console.print(
        f"[bold]Anomalies:[/bold] {stats['anomaly_count']} ({stats['anomaly_pct']:.1f}%)"
    )


@cli.command()
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option(
    "--output",
    "-o",
    default="data/final/tst.jsonl",
    help="Output JSONL path",
)
def export(
    collection: str,
    output: str,
) -> None:
    """Export the enriched dataset (text + quality/cluster/anomaly labels) to JSONL."""
    console.print(
        f"[bold green]Exporting collection {collection} to {output}...[/bold green]"
    )
    n = export_jsonl(collection_name=collection, output_path=output)
    console.print(f"[bold green]Exported {n} records to {output}[/bold green]")


@cli.command()
@click.option(
    "--input",
    "-i",
    default="data/raw/livrointernet12pdf.pdf",
    help="Path to the TST Livro de Jurisprudência PDF",
)
@click.option("--collection", default="amazon_reviews", help="ChromaDB collection name")
@click.option("--min-len", default=20, help="Minimum verbete text length")
@click.option("--skip-ingest", is_flag=True, help="Skip ingestion step")
@click.option("--skip-embed", is_flag=True, help="Skip embedding step")
@click.option("--skip-cluster", is_flag=True, help="Skip clustering step")
@click.option("--skip-quality", is_flag=True, help="Skip quality step")
@click.option("--skip-export", is_flag=True, help="Skip JSONL export step")
def pipeline(
    input: str,
    collection: str,
    min_len: int,
    skip_ingest: bool,
    skip_embed: bool,
    skip_cluster: bool,
    skip_quality: bool,
    skip_export: bool,
) -> None:
    """Run full pipeline: ingest -> embed -> cluster -> quality -> export."""
    if not skip_ingest:
        console.print("[bold blue]=== INGEST ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            ingest,
            input=input,
            output="data/processed/tst.parquet",
            min_len=min_len,
        )
    if not skip_embed:
        console.print("[bold blue]=== EMBED ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            embed,
            input="data/processed/tst.parquet",
            collection=collection,
            batch_size=32,
            device="cpu",
            id_prefix="verbete",
        )
    if not skip_cluster:
        console.print("[bold blue]=== CLUSTER ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(
            cluster,
            collection=collection,
            umap_neighbors=15,
            umap_min_dist=0.1,
            hdbscan_min_cluster=10,
            hdbscan_min_samples=5,
        )
    if not skip_quality:
        console.print("[bold blue]=== QUALITY ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(quality, collection=collection, contamination=0.05)
    if not skip_export:
        console.print("[bold blue]=== EXPORT ===[/bold blue]")
        ctx = click.get_current_context()
        ctx.invoke(export, collection=collection, output="data/final/tst.jsonl")
    console.print("[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    cli()
