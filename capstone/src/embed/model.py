"""Embedding generation using SentenceTransformers."""

# mypy: ignore-errors

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import settings
from store.client import get_chroma_client, get_or_create_collection
from chromadb.api.models.Collection import Collection


def get_model(device: str | None = None) -> SentenceTransformer:
    """Load the embedding model."""
    dev = device or settings.EMBEDDING_DEVICE
    torch.set_num_threads(settings.EMBEDDING_NUM_THREADS)
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=dev)
    return model


def embed_texts(
    texts: list[str], model: SentenceTransformer | None = None, batch_size: int | None = None
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: Pre-loaded SentenceTransformer model (optional)
        batch_size: Batch size for encoding

    Returns:
        List of embedding vectors (list of floats)
    """
    if model is None:
        model = get_model()
    bs = batch_size or settings.EMBEDDING_BATCH_SIZE
    embeddings = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()  # type: ignore[return-value]


def embed_reviews(
    input_path: str,
    collection_name: str,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """
    Load reviews from parquet, generate embeddings, store in ChromaDB.

    Args:
        input_path: Path to parquet file with reviews
        collection_name: ChromaDB collection name
        batch_size: Embedding batch size
        device: Device to use (cpu/cuda)
    """
    console_log(f"Loading reviews from {input_path}...")
    df = pd.read_parquet(input_path)
    console_log(f"Generating embeddings for {len(df)} reviews...")

    model = get_model(device)
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()

    embeddings = embed_texts(texts, model=model, batch_size=batch_size)

    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)

    console_log(f"Storing {len(embeddings)} embeddings in ChromaDB...")
    metadatas = df.to_dict(orient="records")
    ids = [f"review_{i}" for i in range(len(df))]

    for i in range(0, len(ids), settings.CHROMA_BATCH_SIZE):
        end = min(i + settings.CHROMA_BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],  # type: ignore[arg-type]
            metadatas=metadatas[i:end],  # type: ignore[arg-type]
            documents=texts[i:end],
        )
        console_log(
            f"  Stored batch {i//settings.CHROMA_BATCH_SIZE + 1}/{(len(ids)-1)//settings.CHROMA_BATCH_SIZE + 1}"
        )

    console_log("Embedding and storage complete!")


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[EMBED] {msg}")
