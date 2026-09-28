"""Streamlit exploration UI for the document knowledge base."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Resolved from this file so the app works regardless of the current directory:
# streamlit_app.py -> ui -> src -> capstone, and the data lives in capstone/data.
CAPSTONE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CAPSTONE_DATA_DIR", CAPSTONE_ROOT / "data"))
INDEX_DIR = DATA_DIR / "index"
ANALYSIS_FILE = INDEX_DIR / "analysis.json"


@st.cache_resource
def load_analysis() -> Optional[Dict[str, Any]]:
    """Read the analysis produced by the pipeline, cached across reruns."""
    if not ANALYSIS_FILE.exists():
        return None
    with ANALYSIS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_store():
    """Load the vector index for semantic search."""
    from index.vector_store import VectorStore

    try:
        return VectorStore.load(INDEX_DIR)
    except FileNotFoundError:
        return None


def render_sidebar(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Filters that drive the whole exploration view."""
    st.sidebar.header("Filters")

    clusters = sorted({int(value) for value in analysis.get("labels", [])})
    selected_clusters = st.sidebar.multiselect(
        "Clusters",
        options=clusters,
        default=clusters,
        format_func=lambda value: f"Cluster {value}",
    )

    quality_labels = ["high", "medium", "low"]
    selected_quality = st.sidebar.multiselect(
        "Quality", options=quality_labels, default=quality_labels
    )

    hide_anomalies = st.sidebar.checkbox("Hide anomalies", value=False)

    return {
        "clusters": selected_clusters,
        "quality": selected_quality,
        "hide_anomalies": hide_anomalies,
    }


def build_frame(analysis: Dict[str, Any]):
    """Turn the analysis payload into a pandas DataFrame."""
    import pandas as pd

    store = load_store()
    records: List[Dict[str, Any]] = store.metadata if store else []

    rows: List[Dict[str, Any]] = []
    for position, doc_id in enumerate(analysis.get("doc_ids", [])):
        record = records[position] if position < len(records) else {}
        rows.append(
            {
                "doc_id": doc_id,
                "title": record.get("title", doc_id),
                "cluster": analysis["labels"][position],
                "x": analysis["coordinates"][position][0],
                "y": analysis["coordinates"][position][1],
                "quality_score": record.get("quality_score"),
                "quality_label": record.get("quality_label"),
                "is_anomaly": record.get("is_anomaly", False),
            }
        )
    return pd.DataFrame(rows)


def apply_filters(frame, filters: Dict[str, Any]):
    """Filter the exploration frame by sidebar selections."""
    import pandas as pd

    mask = frame["cluster"].isin(filters["clusters"])
    mask &= frame["quality_label"].isin(filters["quality"])
    if filters["hide_anomalies"]:
        mask &= ~frame["is_anomaly"].fillna(False).astype(bool)
    return frame[mask]


def page_overview(analysis: Dict[str, Any]) -> None:
    """Corpus level metrics."""
    st.subheader("Corpus overview")
    summary = analysis.get("analysis", {})
    quality = summary.get("quality", {})
    anomalies = summary.get("anomalies", {})

    columns = st.columns(4)
    columns[0].metric("Documents", summary.get("documents", 0))
    columns[1].metric("Clusters", len(summary.get("clusters", {})))
    columns[2].metric("Mean quality", f"{quality.get('mean_score', 0.0):.2f}")
    columns[3].metric("Anomalies", anomalies.get("anomalies", 0))

    st.caption(
        f"Backend `{summary.get('backend', 'n/a')}` · "
        f"{summary.get('dimension', 0)}-dimensional vectors · "
        f"clustered with `{summary.get('algorithm', 'n/a')}` · "
        f"projected with `{summary.get('projection', 'n/a')}`"
    )
    variance = summary.get("variance_explained")
    if variance is not None:
        st.caption(
            f"The two axes keep {float(variance) * 100:.1f}% of the original "
            "variance, so treat distances on the map as an approximation."
        )

    if summary.get("clusters"):
        st.markdown("**Documents per cluster**")
        st.bar_chart(summary["clusters"])

    if quality.get("distribution"):
        st.markdown("**Quality distribution**")
        st.bar_chart(quality["distribution"])


def page_explore(analysis: Dict[str, Any], frame) -> None:
    """Interactive 2D cluster map."""
    import plotly.express as px

    st.subheader("Cluster map")
    st.caption(
        "Each point is a paper. Papers close together have similar embeddings, "
        "so proximity means similar content. Axes come from PCA over the vectors."
    )

    hover = frame[["doc_id", "title", "quality_label"]].to_dict("records")
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        symbol="quality_label",
        hover_name="title",
        hover_data={"quality_label": True, "doc_id": False, "x": False, "y": False},
        color_continuous_scale="Viridis" if frame["cluster"].nunique() > 8 else None,
    )
    figure.update_traces(marker={"size": 8, "opacity": 0.75, "line": {"width": 0}})
    figure.update_layout(height=620, margin={"t": 10})
    st.plotly_chart(figure, width="stretch")

    cluster_sizes = frame.groupby("cluster").size().sort_values(ascending=False)
    selected = st.selectbox(
        "Inspect a cluster", options=[int(c) for c in cluster_sizes.index]
    )
    members = frame[frame["cluster"] == selected].sort_values(
        "quality_score", ascending=False
    )
    st.markdown(
        f"**Cluster {selected}** — {len(members)} papers, highest quality first"
    )
    st.dataframe(
        members[["title", "quality_label", "quality_score", "is_anomaly"]],
        width="stretch",
        hide_index=True,
    )


def page_search() -> None:
    """Semantic search over the index."""
    from index.search import SemanticSearch, load_embedder

    st.subheader("Semantic search")
    st.caption(
        "Describe what you are looking for in plain language. Results come from the "
        "vector index, not from keyword matching."
    )

    store = load_store()
    if store is None:
        st.warning(
            "No index found. Run: capstone ingest && capstone build && capstone cluster"
        )
        return

    with st.form("search_form"):
        query = st.text_input(
            "Query", placeholder="how do neural networks learn representations"
        )
        backends = ["tfidf", "minilm"]
        default_backend = store.backend if store.backend in backends else "tfidf"
        backend = st.selectbox(
            "Embedding backend",
            options=backends,
            index=backends.index(default_backend),
            help="Must match the backend used to build the index.",
        )
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search")

    if not submitted or not query:
        return

    with st.spinner("Searching..."):
        try:
            embedder = load_embedder(backend, DATA_DIR)
            hits = SemanticSearch(store, embedder=embedder).query(query, k=top_k)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return

    for rank, hit in enumerate(hits, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {hit.title}**")
            st.caption(f"similarity {hit.score:.3f} · {hit.doc_id}")
            if "text" in hit.metadata:
                st.text(hit.metadata["text"][:600])


def page_quality(analysis: Dict[str, Any], frame) -> None:
    """Quality control and anomaly inspection."""
    st.subheader("Quality control")
    st.caption(
        "Quality scores come from transparent rules tuned for scientific papers: enough "
        "content, expected sections and readable sentences. Anomalies are documents whose "
        "embedding is unusually far from the rest of the corpus."
    )

    low_quality = frame[frame["quality_label"] == "low"].sort_values("quality_score")
    st.markdown("**Lowest quality documents**")
    st.dataframe(
        low_quality[["title", "quality_score", "quality_label", "cluster"]].head(25),
        width="stretch",
        hide_index=True,
    )

    anomalies = frame[frame["is_anomaly"].fillna(False).astype(bool)]
    st.markdown(f"**Anomalies ({len(anomalies)})**")
    if len(anomalies):
        st.dataframe(
            anomalies[["title", "cluster", "quality_score"]].head(25),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No anomalies detected with the current threshold.")


def main() -> None:
    st.set_page_config(page_title="Document Navigator", layout="wide")
    st.title("Document Navigator")
    st.caption("Exploring a corpus of scientific papers with embeddings and clustering")

    analysis = load_analysis()
    if analysis is None:
        st.error(
            "No analysis available. Build the knowledge base first:\n\n"
            "```bash\n"
            "capstone ingest --max-results 500\n"
            "capstone build --backend tfidf\n"
            "capstone cluster\n"
            "```"
        )
        return

    try:
        frame = build_frame(analysis)
    except (KeyError, TypeError, ValueError) as exc:
        st.error(
            f"The analysis file could not be read ({exc}). "
            "Re-run 'capstone cluster' to regenerate it."
        )
        return

    if frame.empty:
        st.warning("The analysis contains no documents. Re-run 'capstone cluster'.")
        return

    filters = render_sidebar(analysis)
    visible = apply_filters(frame, filters)

    st.caption(f"Showing {len(visible)} of {len(frame)} documents")

    page_map, page_find, page_over, page_quality_tab = st.tabs(
        ["Explore", "Semantic search", "Overview", "Quality"]
    )

    with page_map:
        page_explore(analysis, visible)
    with page_find:
        page_search()
    with page_over:
        page_overview(analysis)
    with page_quality_tab:
        page_quality(analysis, visible)


if __name__ == "__main__":
    main()
