"""End-to-end pipeline orchestrating ingest, embed, cluster and score."""

from .core import ANALYSIS_FILE, TFIDF_MODEL_FILE, Pipeline, PipelinePaths

__all__ = ["Pipeline", "PipelinePaths", "ANALYSIS_FILE", "TFIDF_MODEL_FILE"]
