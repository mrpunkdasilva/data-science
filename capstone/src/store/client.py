"""ChromaDB client and collection management."""

# mypy: ignore-errors

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection
from config import settings


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB persistent client."""
    client = chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient, name: str) -> Collection:
    """Get existing collection or create new one with HNSW config."""
    try:
        collection = client.get_collection(name)
        console_log(f"Using existing collection: {name}")
    except Exception:
        collection = client.create_collection(
            name=name,
            metadata={
                "hnsw:space": settings.HNSW_SPACE,
                "hnsw:M": settings.HNSW_M,
                "hnsw:construction_ef": settings.HNSW_EF_CONSTRUCTION,
            },
        )
        console_log(f"Created new collection: {name}")
    return collection


def get_collection(name: str) -> Collection:
    """Get existing collection (raises if not exists)."""
    client = get_chroma_client()
    return client.get_collection(name)  # type: ignore[return-value]


def delete_collection(name: str) -> None:
    """Delete a collection entirely."""
    client = get_chroma_client()
    client.delete_collection(name)
    console_log(f"Deleted collection: {name}")


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[STORE] {msg}")
