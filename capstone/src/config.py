"""Centralized configuration for the capstone pipeline."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    FINAL_DIR: Path = DATA_DIR / "final"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NUM_THREADS: int = 8

    # Source - TST Livro de Jurisprudência (Súmulas, OJs, Precedentes Normativos)
    TST_PDF_PATH: Path = RAW_DIR / "livrointernet12pdf.pdf"
    TST_PDF_URL: str = "https://www.tst.jus.br/livro-de-sumulas-ojs-e-pns"
    VERBETE_MIN_TEXT: int = 20
    VERBETE_CURTO: int = 120
    VERBETE_RUIDO: int = 30

    # ChromaDB
    COLLECTION_NAME: str = "amazon_reviews"
    HNSW_SPACE: str = "cosine"
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 200
    CHROMA_BATCH_SIZE: int = 500

    # Clustering (UMAP + HDBSCAN)
    UMAP_N_NEIGHBORS: int = 15
    UMAP_MIN_DIST: float = 0.1
    UMAP_N_COMPONENTS: int = 2
    UMAP_METRIC: str = "cosine"
    UMAP_RANDOM_STATE: int = 42
    HDBSCAN_MIN_CLUSTER_SIZE: int = 15
    HDBSCAN_MIN_SAMPLES: int = 5
    HDBSCAN_METRIC: str = "euclidean"
    HDBSCAN_CLUSTER_SELECTION_EPSILON: float = 0.0

    # Quality Control
    CLASSIFIER_TEST_SIZE: float = 0.2
    CLASSIFIER_RANDOM_STATE: int = 42
    CLASSIFIER_MAX_FEATURES: int = 5000
    CLASSIFIER_NGRAM_RANGE: tuple[int, int] = (1, 2)
    ANOMALY_CONTAMINATION: float = 0.05
    ANOMALY_N_ESTIMATORS: int = 200
    ANOMALY_RANDOM_STATE: int = 42

    # Search
    DEFAULT_SEARCH_K: int = 10
    MAX_SEARCH_K: int = 50

    # Export
    EXPORT_PATH: Path = FINAL_DIR / "tst.jsonl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
for path in [
    settings.RAW_DIR,
    settings.PROCESSED_DIR,
    settings.FINAL_DIR,
    settings.MODELS_DIR,
    settings.CHROMA_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
