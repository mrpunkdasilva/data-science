"""End-to-end pipeline: ingest, embed, cluster and score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from cluster.clustering import Clusterer
from cluster.dim_reduction import DimReducer
from embeddings.factory import embed_corpus
from ingest.loader import Document, DocumentLoader
from quality.anomalies import AnomalyDetector
from quality.classifier import QualityClassifier
from index.vector_store import VectorStore

ANALYSIS_FILE = "analysis.json"
TFIDF_MODEL_FILE = "tfidf_svd.joblib"


@dataclass
class PipelinePaths:
    """Where the pipeline reads and writes."""

    corpus_dir: Path
    index_dir: Path
    models_dir: Path

    @classmethod
    def under(cls, root: Path) -> "PipelinePaths":
        root = Path(root)
        return cls(
            corpus_dir=root / "corpus",
            index_dir=root / "index",
            models_dir=root / "models",
        )


class Pipeline:
    """
    Runs the four stages of the project in order.

    Ingestion loads the documents, embedding turns them into vectors, the
    index is persisted for search, then clustering, quality scoring and anomaly
    detection enrich every document with its cluster and its scores.
    """

    def __init__(
        self,
        paths: PipelinePaths,
        backend: str = "tfidf",
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.paths = paths
        self.backend = backend
        self.model_name = model_name
        self.batch_size = batch_size

    def load_documents(self) -> List[Document]:
        return DocumentLoader(self.paths.corpus_dir).load_all()

    def build_index(
        self, documents: List[Document]
    ) -> tuple[VectorStore, Dict[str, Any]]:
        """Embed the documents and persist the vector index."""
        texts = [document.text for document in documents]

        if self.backend == "tfidf":
            from embeddings.tfidf_svd import TfidfSvdEmbedder

            embedder = TfidfSvdEmbedder()
            matrix = embedder.fit_transform(texts)
            backend_name = embedder.name
            dimension = embedder.dimension
            # Persist the fitted model so search projects queries in the same space
            embedder.save(self.paths.models_dir / TFIDF_MODEL_FILE)
        else:
            matrix, backend_name, dimension = embed_corpus(
                texts,
                backend=self.backend,
                model_name=self.model_name,
                batch_size=self.batch_size,
            )

        store = VectorStore(
            self.paths.index_dir, dimension=dimension, backend=backend_name
        )
        store.build(
            doc_ids=[document.doc_id for document in documents],
            vectors=matrix,
            metadata=[
                document.metadata
                | {
                    "title": document.title,
                    "source_type": document.source_type,
                    "text": document.text,
                }
                for document in documents
            ],
        )
        store.save()
        return store, {
            "backend": backend_name,
            "dimension": dimension,
            "documents": len(documents),
        }

    def analyze(
        self,
        store: VectorStore,
        algorithm: str = "kmeans",
        n_clusters: int = 8,
        projection: str = "pca",
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """Cluster the documents, project them to 2D and score their quality."""
        vectors = store.vectors
        doc_ids = store.doc_ids
        metadata = store.metadata

        reducer = DimReducer(method=projection)
        coordinates = reducer.fit_transform(vectors)

        clusterer = Clusterer(algorithm=algorithm, n_clusters=n_clusters)
        labels = clusterer.fit_predict(vectors)

        classifier = QualityClassifier()
        quality = classifier.score_documents(
            [str(record.get("text", "")) for record in metadata],
            [str(record.get("title", "")) for record in metadata],
        )

        detector = AnomalyDetector(
            method="isolation_forest", contamination=contamination
        )
        anomalies = detector.fit_predict(vectors)

        for position, doc_id in enumerate(doc_ids):
            metadata[position]["cluster"] = int(labels[position])
            metadata[position]["x"] = float(coordinates[position][0])
            metadata[position]["y"] = float(coordinates[position][1])
            metadata[position]["quality_score"] = float(quality["scores"][position])
            metadata[position]["quality_label"] = quality["labels"][position]
            metadata[position]["is_anomaly"] = bool(anomalies[position])

        analysis = {
            "documents": len(doc_ids),
            "dimension": store.dimension,
            "backend": store.backend,
            "algorithm": algorithm,
            "projection": projection,
            "variance_explained": reducer.explains_variance,
            "n_clusters": int(len({int(value) for value in labels})),
            "clusters": {
                str(label): int((labels == label).sum())
                for label in sorted({int(value) for value in labels})
            },
            "quality": classifier.summary(quality),
            "anomalies": detector.summary(doc_ids),
        }
        self._save_analysis(analysis, labels, coordinates, doc_ids)
        store.save()
        return analysis

    def _save_analysis(
        self,
        analysis: Dict[str, Any],
        labels: NDArray[np.int64],
        coordinates: NDArray[np.float32],
        doc_ids: List[str],
    ) -> None:
        """Persist cluster assignments and 2D coordinates for the UI."""
        self.paths.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "analysis": analysis,
            "doc_ids": doc_ids,
            "labels": labels.tolist(),
            "coordinates": coordinates.tolist(),
        }
        with (self.paths.index_dir / ANALYSIS_FILE).open(
            "w", encoding="utf-8"
        ) as handle:
            # allow_nan=False makes a stray NaN an error instead of the invalid
            # `NaN` token, which only Python's own json module can read back.
            json.dump(payload, handle, ensure_ascii=False, indent=1, allow_nan=False)
