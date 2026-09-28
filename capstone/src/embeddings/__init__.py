"""Text embedding backends: neural model and lexical fallback."""

from .base import EmbeddingModel, cosine_similarity, l2_normalize
from .factory import BACKENDS, build_embedder, embed_corpus
from .tfidf_svd import TfidfSvdEmbedder

__all__ = [
    "EmbeddingModel",
    "cosine_similarity",
    "l2_normalize",
    "BACKENDS",
    "build_embedder",
    "embed_corpus",
    "TfidfSvdEmbedder",
]
