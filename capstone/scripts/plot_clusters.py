"""Generate the cluster map used in the capstone presentation slide."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ANALYSIS = Path("capstone/data/index/analysis.json")
OUTPUT = Path("documentation/images/capstone-clustering.png")

# How each projection names its two axes, and how to describe it in the title.
AXIS_LABELS = {"pca": ("PC 1", "PC 2"), "umap": ("UMAP 1", "UMAP 2")}
PROJECTION_NAMES = {"pca": "PCA", "umap": "UMAP", "none": "no projection"}


def main() -> None:
    with ANALYSIS.open(encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    coordinates: NDArray[np.float32] = np.asarray(
        payload["coordinates"], dtype=np.float32
    )
    labels: NDArray[np.int64] = np.asarray(payload["labels"], dtype=np.int64)
    doc_ids = payload["doc_ids"]

    # Read the settings from the analysis instead of hardcoding them, so the
    # caption can never claim a projection the figure was not built with.
    summary = payload.get("analysis", {})
    projection = str(summary.get("projection", "pca")).lower()
    backend = str(summary.get("backend", "tfidf"))
    backend_name = "TF-IDF + SVD" if backend == "tfidf" else "MiniLM"
    x_label, y_label = AXIS_LABELS.get(projection, ("dim 1", "dim 2"))
    projection_name = PROJECTION_NAMES.get(projection, projection)

    figure, axes = plt.subplots(figsize=(11, 5.6), dpi=200)
    cmap = plt.get_cmap("tab10")

    unique_labels: List[int] = [int(value) for value in np.unique(labels)]
    for cluster in unique_labels:
        mask = labels == cluster
        axes.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=26,
            alpha=0.72,
            color=cmap(cluster % 10),
            edgecolors="none",
            label=f"Cluster {cluster} (n={int(mask.sum())})",
        )

    axes.set_title(
        f"arXiv science corpus: {len(doc_ids)} papers, "
        f"{len(unique_labels)} clusters, {projection_name} over {backend_name}",
        fontsize=11,
        pad=12,
    )
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(alpha=0.18, linewidth=0.6)
    axes.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=1.5,
    )
    figure.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(
        f"Wrote {OUTPUT} for {len(doc_ids)} documents, "
        f"{len(unique_labels)} clusters"
    )


if __name__ == "__main__":
    main()
