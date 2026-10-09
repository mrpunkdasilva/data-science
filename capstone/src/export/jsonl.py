"""Export the enriched verbete dataset to JSONL (one JSON object per line)."""

# mypy: ignore-errors

import json
from pathlib import Path
from typing import Any

from store.client import get_chroma_client, get_collection


def _build_record(
    chroma_id: str,
    metadata: dict[str, Any],
    document: str,
) -> dict[str, Any]:
    """Build a single JSONL record from a ChromaDB entry."""
    record: dict[str, Any] = {
        "id": chroma_id,
        "doc_id": metadata.get("doc_id", chroma_id),
        "tipo": metadata.get("tipo", ""),
        "orgao": metadata.get("orgao", ""),
        "codigo": metadata.get("codigo", ""),
        "tema": metadata.get("tema", ""),
        "text": metadata.get("text", document),
        "text_len": int(metadata.get("text_len", len(metadata.get("text", document)))),
        "source": metadata.get("source", ""),
        "quality_label": metadata.get("quality_label", "completo"),
        "cluster_id": int(metadata.get("cluster_id", -1)),
        "anomaly_score": round(float(metadata.get("anomaly_score", 0.0)), 6),
    }
    if "umap_x" in metadata and "umap_y" in metadata:
        record["umap_x"] = round(float(metadata["umap_x"]), 6)
        record["umap_y"] = round(float(metadata["umap_y"]), 6)
    return record


def export_jsonl(
    collection_name: str = "amazon_reviews",
    output_path: str = "data/final/tst.jsonl",
) -> int:
    """
    Export the enriched verbete dataset (text + quality/cluster/anomaly labels)
    from ChromaDB to a JSONL file.

    Args:
        collection_name: ChromaDB collection name
        output_path: Path of the output JSONL file

    Returns:
        Number of records written
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    count = collection.count()
    if count == 0:
        console_log("Collection is empty, nothing to export!")
        return 0

    result = collection.get(include=["metadatas", "documents"], limit=count)  # type: ignore[arg-type]
    ids = result["ids"] or []
    metadatas = result["metadatas"] or []
    documents = result["documents"] or []

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for chroma_id, meta, doc in zip(ids, metadatas, documents):
            fh.write(
                json.dumps(_build_record(chroma_id, meta, doc), ensure_ascii=False)
            )
            fh.write("\n")
            written += 1

    console_log(f"Exported {written} records to {output_path}")
    return written


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[EXPORT] {msg}")
