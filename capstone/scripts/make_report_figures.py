"""Generate the figures and summary used by the capstone reports.

Reads the pipeline artifacts (``data/processed/tst.parquet`` with the heuristic
labels and ``data/final/tst.jsonl`` with the enriched labels) and writes:

- ``docs/figures/*.png`` figure files embedded in the reports.
- ``docs/figures/summary.json`` machine-readable summary of all metrics.

Run it with ``make docs`` or directly::

    python scripts/make_report_figures.py
"""

# mypy: ignore-errors

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "tst.parquet"
FINAL = ROOT / "data" / "final" / "tst.jsonl"
FIGURES = ROOT / "docs" / "figures"

QUALITY_COLORS = {
    "completo": "#2e7d32",
    "cancelado": "#c62828",
    "curto": "#ef6c00",
    "duplicado": "#6a1b9a",
    "ruido": "#9e9e9e",
}

ANOMALY_THRESHOLD = 0.7
CLASSIFIER_TEST_SIZE = 0.2
CLASSIFIER_RANDOM_STATE = 42
CLASSIFIER_MAX_FEATURES = 5000
CLASSIFIER_NGRAM_RANGE = (1, 2)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the heuristic (parquet) and enriched (jsonl) datasets."""
    if not PROCESSED.exists():
        raise SystemExit(f"Missing {PROCESSED}. Run `make pipeline` first.")
    if not FINAL.exists():
        raise SystemExit(f"Missing {FINAL}. Run `make pipeline` first.")
    heuristic = pd.read_parquet(PROCESSED)
    enriched = pd.DataFrame(
        [
            json.loads(line)
            for line in FINAL.read_text(encoding="utf-8").splitlines()
            if line
        ]
    )
    return heuristic, enriched


def classifier_metrics(heuristic: pd.DataFrame) -> dict:
    """Reproduce the quality classifier evaluation from the pipeline."""
    texts = (heuristic["tema"].fillna("") + " " + heuristic["text"].fillna("")).tolist()
    labels = heuristic["quality_label"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=CLASSIFIER_MAX_FEATURES,
        ngram_range=CLASSIFIER_NGRAM_RANGE,
        min_df=2,
        max_df=0.95,
    )
    features = vectorizer.fit_transform(texts)

    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=CLASSIFIER_TEST_SIZE,
        random_state=CLASSIFIER_RANDOM_STATE,
        stratify=stratify,
    )
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=CLASSIFIER_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    from sklearn.metrics import classification_report

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "report_text": classification_report(y_test, y_pred),
    }


def cluster_top_terms(enriched: pd.DataFrame, cluster_id: int, n: int = 8) -> list[str]:
    """Most frequent words inside a cluster."""
    texts = enriched.loc[enriched["cluster_id"] == cluster_id, "text"].astype(str)
    words = [w for text in texts for w in text.lower().split() if len(w) > 3]
    return [w for w, _ in Counter(words).most_common(n)]


def figure_quality_labels(heuristic: pd.DataFrame, enriched: pd.DataFrame) -> None:
    """Grouped bar chart comparing heuristic and predicted quality labels."""
    order = ["completo", "cancelado", "curto", "duplicado"]
    h = Counter(heuristic["quality_label"])
    p = Counter(enriched["quality_label"])
    x = np.arange(len(order))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, [h.get(k, 0) for k in order], width, label="heuristic")
    ax.bar(x + width / 2, [p.get(k, 0) for k in order], width, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("documents")
    ax.set_title("Quality labels: heuristic vs. classifier prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_quality_labels.png", dpi=150)
    plt.close(fig)


def figure_anomaly_scores(enriched: pd.DataFrame) -> None:
    """Histogram of IsolationForest anomaly scores."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(enriched["anomaly_score"], bins=40, color="#1565c0")
    ax.axvline(
        ANOMALY_THRESHOLD, color="#c62828", linestyle="--", label="threshold 0.7"
    )
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("documents")
    ax.set_title("IsolationForest anomaly score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_anomaly_scores.png", dpi=150)
    plt.close(fig)


def figure_cluster_map(enriched: pd.DataFrame) -> None:
    """UMAP scatter coloured by HDBSCAN cluster."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in sorted(enriched["cluster_id"].unique()):
        subset = enriched[enriched["cluster_id"] == cluster_id]
        label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        color = "#9e9e9e" if cluster_id == -1 else None
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=12,
            alpha=0.7,
            label=f"{label} (n={len(subset)})",
            color=color,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP projection with HDBSCAN clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cluster_map.png", dpi=150)
    plt.close(fig)


def figure_cluster_sizes(enriched: pd.DataFrame) -> None:
    """Bar chart of cluster sizes (noise excluded)."""
    sizes = (
        enriched[enriched["cluster_id"] != -1]
        .groupby("cluster_id")
        .size()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([f"cluster {int(c)}" for c in sizes.index], sizes.values, color="#00838f")
    ax.set_ylabel("documents")
    ax.set_title("Cluster sizes (noise excluded)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cluster_sizes.png", dpi=150)
    plt.close(fig)


def figure_tipo_distribution(enriched: pd.DataFrame) -> None:
    """Bar chart of documents by tipo."""
    counts = enriched["tipo"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color="#5e35b1")
    ax.set_ylabel("documents")
    ax.set_title("Documents by tipo")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_tipo.png", dpi=150)
    plt.close(fig)


def figure_text_length(enriched: pd.DataFrame) -> None:
    """Histogram of verbete text length."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(enriched["text_len"], bins=40, color="#ef6c00")
    ax.set_xlabel("characters")
    ax.set_ylabel("documents")
    ax.set_title("Verbete text length distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_text_length.png", dpi=150)
    plt.close(fig)


def figure_architecture() -> None:
    """Box-and-arrow architecture diagram for the technical report."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor=color, edgecolor="#263238")
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#37474f"),
        )

    box(0.0, 3.2, 1.5, 0.9, "TST PDF\n(offline source)", "#e3f2fd")
    box(2.0, 3.2, 1.5, 0.9, "ingest\ntst.py", "#bbdefb")
    box(4.0, 3.2, 1.5, 0.9, "parquet\n1058 verbetes", "#bbdefb")
    box(6.0, 3.2, 1.6, 0.9, "embed\nMiniLM 384d", "#bbdefb")
    box(8.0, 3.2, 1.6, 0.9, "ChromaDB\nHNSW cosine", "#ffe0b2")
    box(8.0, 1.6, 1.6, 0.9, "quality\nTF-IDF + LogReg\nIsolationForest", "#c8e6c9")
    box(6.0, 1.6, 1.6, 0.9, "cluster\nUMAP + HDBSCAN", "#c8e6c9")
    box(3.0, 0.2, 1.8, 0.9, "export\nJSONL", "#f8bbd0")
    box(0.5, 0.2, 2.0, 0.9, "dashboard\nStreamlit + Plotly", "#f0f4c3")
    box(1.5, 1.6, 1.3, 0.9, "CLI\nclick", "#f0f4c3")

    arrow(1.5, 3.65, 2.0, 3.65)
    arrow(3.5, 3.65, 4.0, 3.65)
    arrow(5.5, 3.65, 6.0, 3.65)
    arrow(7.6, 3.65, 8.0, 3.65)
    arrow(8.8, 3.2, 8.8, 2.5)
    arrow(8.0, 2.05, 7.6, 2.05)
    arrow(6.8, 1.6, 4.5, 1.1)
    arrow(8.0, 1.6, 4.5, 1.1)
    arrow(3.0, 0.65, 2.5, 0.65)
    arrow(1.5, 2.05, 1.0, 1.1)
    arrow(9.6, 3.2, 9.6, 1.1)

    ax.set_xlim(-0.2, 10.0)
    ax.set_ylim(-0.1, 4.4)
    ax.set_title("End-to-end pipeline architecture")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_architecture.png", dpi=150)
    plt.close(fig)


def build_summary(heuristic: pd.DataFrame, enriched: pd.DataFrame) -> dict:
    """Assemble the metrics dictionary written to summary.json."""
    metrics = classifier_metrics(heuristic)
    top_anomalies = (
        enriched.sort_values("anomaly_score", ascending=False)
        .head(10)[["doc_id", "tipo", "orgao", "quality_label", "anomaly_score", "text"]]
        .to_dict("records")
    )
    for row in top_anomalies:
        row["text"] = str(row["text"])[:280]
        row["anomaly_score"] = round(float(row["anomaly_score"]), 4)

    cluster_sizes = (
        enriched[enriched["cluster_id"] != -1]
        .groupby("cluster_id")
        .size()
        .sort_values(ascending=False)
    )
    clusters = [
        {
            "cluster_id": int(cid),
            "size": int(size),
            "top_terms": cluster_top_terms(enriched, int(cid)),
        }
        for cid, size in cluster_sizes.items()
    ]

    return {
        "total_documents": int(len(enriched)),
        "heuristic_quality": dict(Counter(heuristic["quality_label"])),
        "predicted_quality": dict(Counter(enriched["quality_label"])),
        "tipos": dict(Counter(enriched["tipo"])),
        "text_len": {
            "min": int(enriched["text_len"].min()),
            "median": int(enriched["text_len"].median()),
            "max": int(enriched["text_len"].max()),
        },
        "n_clusters": int(len(cluster_sizes)),
        "noise_count": int((enriched["cluster_id"] == -1).sum()),
        "clusters": clusters,
        "anomaly_count_gt_0_7": int(
            (enriched["anomaly_score"] > ANOMALY_THRESHOLD).sum()
        ),
        "anomaly_mean": round(float(enriched["anomaly_score"].mean()), 4),
        "top_anomalies": top_anomalies,
        "classifier": metrics,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    heuristic, enriched = load_data()

    figure_quality_labels(heuristic, enriched)
    figure_anomaly_scores(enriched)
    figure_cluster_map(enriched)
    figure_cluster_sizes(enriched)
    figure_tipo_distribution(enriched)
    figure_text_length(enriched)
    figure_architecture()

    summary = build_summary(heuristic, enriched)
    (FIGURES / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Figures written to {FIGURES}")
    print(f"Total documents: {summary['total_documents']}")
    print(f"Heuristic quality: {summary['heuristic_quality']}")
    print(f"Predicted quality: {summary['predicted_quality']}")
    print(f"Tipos: {summary['tipos']}")
    print(f"Clusters: {summary['n_clusters']} (noise={summary['noise_count']})")
    print(f"Anomalies > 0.7: {summary['anomaly_count_gt_0_7']}")
    print(f"Classifier accuracy: {summary['classifier']['accuracy']:.4f}")
    print(f"Classifier F1 macro: {summary['classifier']['f1_macro']:.4f}")
    print()
    print(summary["classifier"]["report_text"])


if __name__ == "__main__":
    main()
