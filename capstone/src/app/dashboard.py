"""Interactive Streamlit dashboard for the TST Jurisprudence pipeline.

Launch it with:

    streamlit run src/app/dashboard.py

The dashboard reads the enriched dataset produced by the pipeline export step
(``data/final/tst.jsonl``) and offers three views:

- Semantic search over the verbetes (ChromaDB, with a keyword fallback).
- An interactive UMAP cluster map built with Plotly.
- A quality control and anomaly panel with distributions and flagged documents.
"""

# mypy: ignore-errors

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings

NOISE_CLUSTER = -1

QUALITY_COLORS = {
    "completo": "#2e7d32",
    "cancelado": "#c62828",
    "curto": "#ef6c00",
    "duplicado": "#6a1b9a",
    "ruido": "#9e9e9e",
}

CLUSTER_COLORS = {
    "noise": "#9e9e9e",
}


@st.cache_data(show_spinner=False)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    """Load the enriched JSONL export into a DataFrame (cached by mtime)."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    frame = pd.DataFrame(records)
    frame["cluster_id"] = frame["cluster_id"].astype(int)
    frame["cluster_label"] = frame["cluster_id"].apply(
        lambda cid: "noise" if cid == NOISE_CLUSTER else f"cluster {cid}"
    )
    return frame


def load_data() -> pd.DataFrame | None:
    """Load the dataset, warning the user when the export is missing."""
    path = Path(settings.EXPORT_PATH)
    if not path.exists():
        st.warning(
            f"Dataset not found at `{path}`. Run `make pipeline` first to ingest, "
            "embed, cluster and export the verbetes."
        )
        return None
    return load_dataset(str(path), path.stat().st_mtime)


def top_terms(frame: pd.DataFrame, cluster_id: int, n: int = 8) -> list[str]:
    """Return the most frequent words inside a cluster (simple descriptor)."""
    texts = frame.loc[frame["cluster_id"] == cluster_id, "text"].astype(str)
    words = [w for text in texts for w in text.lower().split() if len(w) > 3]
    return [word for word, _ in Counter(words).most_common(n)]


@st.cache_resource(show_spinner=False)
def collection_available() -> bool:
    """Check whether the ChromaDB collection required for search exists."""
    try:
        from store.client import get_collection

        get_collection(settings.COLLECTION_NAME)
        return True
    except Exception:
        return False


def semantic_search(
    query: str,
    k: int,
    tipo: str | None,
    orgao: str | None,
    quality: str | None,
) -> list[dict]:
    """Run a semantic search against ChromaDB."""
    from search.engine import search_reviews

    return search_reviews(
        query=query,
        k=k,
        collection_name=settings.COLLECTION_NAME,
        tipo=tipo,
        orgao=orgao,
        quality_label=quality,
    )


def lexical_search(
    frame: pd.DataFrame,
    query: str,
    k: int,
    tipo: str | None,
    orgao: str | None,
    quality: str | None,
) -> list[dict]:
    """Keyword fallback used when the vector store is unavailable."""
    filtered = frame
    if tipo:
        filtered = filtered[filtered["tipo"] == tipo]
    if orgao:
        filtered = filtered[filtered["orgao"] == orgao]
    if quality:
        filtered = filtered[filtered["quality_label"] == quality]

    tokens = [token for token in query.lower().split() if token]

    def score(text: str) -> int:
        low = str(text).lower()
        return sum(low.count(token) for token in tokens)

    scored = filtered.assign(score=filtered["text"].map(score))
    scored = scored[scored["score"] > 0].sort_values("score", ascending=False)
    return scored.head(k)[
        ["tema", "tipo", "orgao", "codigo", "doc_id", "text", "score"]
    ].to_dict("records")


def render_search(frame: pd.DataFrame) -> None:
    """Render the semantic search tab."""
    st.subheader("Semantic search")
    st.caption(
        "Search the verbetes by meaning. Results are ranked by cosine similarity "
        "in ChromaDB when the vector store is available."
    )

    query = st.text_input(
        "Query",
        placeholder="e.g. intervalo intrajornada, adicional de insalubridade",
    )

    col_tipo, col_orgao, col_quality = st.columns(3)
    tipo = col_tipo.selectbox("Tipo", ["", *sorted(frame["tipo"].dropna().unique())])
    orgao = col_orgao.selectbox(
        "Organ", ["", *sorted(frame["orgao"].dropna().unique())]
    )
    quality = col_quality.selectbox(
        "Quality", ["", *sorted(frame["quality_label"].dropna().unique())]
    )
    k = st.slider("Results", min_value=1, max_value=20, value=5)

    if not query:
        st.info("Type a query above to search the verbetes.")
        return

    tipo_arg = tipo or None
    orgao_arg = orgao or None
    quality_arg = quality or None

    if collection_available():
        st.caption("Ranking source: ChromaDB cosine similarity.")
        results = semantic_search(query, k, tipo_arg, orgao_arg, quality_arg)
    else:
        st.caption("ChromaDB unavailable; using keyword fallback over the export.")
        results = lexical_search(frame, query, k, tipo_arg, orgao_arg, quality_arg)

    if not results:
        st.warning("No results found for this query and filter combination.")
        return

    for result in results:
        with st.container(border=True):
            st.markdown(
                f"**{result['tema'] or '(no tema)'}** "
                f"- score `{float(result['score']):.3f}`"
            )
            st.caption(
                f"{result['tipo']} | {result['orgao']} | "
                f"{result['codigo']} | {result['doc_id']}"
            )
            st.write(str(result["text"])[:600])


def render_clusters(frame: pd.DataFrame) -> None:
    """Render the interactive UMAP cluster map tab."""
    st.subheader("Interactive cluster map")
    st.caption(
        "UMAP projection of the 384-dimensional embeddings, coloured by the "
        "HDBSCAN cluster. Hover a point to inspect the verbete."
    )

    labeled = frame.dropna(subset=["umap_x", "umap_y"])
    if labeled.empty:
        st.warning("No UMAP coordinates found. Run `make cluster` to generate them.")
        return

    options = ["all", *sorted(labeled["cluster_label"].unique())]
    selected = st.multiselect("Clusters to display", options, default=options)
    if not selected:
        st.info("Select at least one cluster to display the map.")
        return

    view = (
        labeled[labeled["cluster_label"].isin(selected)]
        if "all" not in selected
        else labeled
    )

    figure = px.scatter(
        view,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        hover_data={
            "tema": True,
            "tipo": True,
            "orgao": True,
            "quality_label": True,
            "anomaly_score": ":.3f",
            "umap_x": False,
            "umap_y": False,
            "cluster_label": False,
        },
        color_discrete_map=CLUSTER_COLORS,
    )
    figure.update_traces(marker={"size": 7, "opacity": 0.75})
    figure.update_layout(height=650, legend_title="Cluster")
    st.plotly_chart(figure, use_container_width=True)

    sizes = (
        labeled[labeled["cluster_id"] != NOISE_CLUSTER]
        .groupby("cluster_id")
        .size()
        .rename("verbetes")
        .reset_index()
        .sort_values("verbetes", ascending=False)
    )
    if sizes.empty:
        st.info(
            "No dense clusters were found (all points are noise). This can happen "
            "with a very small dataset."
        )
        return

    figure_sizes = px.bar(
        sizes,
        x="cluster_id",
        y="verbetes",
        title="Cluster sizes (noise excluded)",
    )
    st.plotly_chart(figure_sizes, use_container_width=True)

    st.markdown("#### Cluster profiles")
    for cluster_id in sizes["cluster_id"]:
        subset = frame[frame["cluster_id"] == cluster_id]
        dominant = subset["quality_label"].value_counts().idxmax()
        tipo_counts = subset["tipo"].value_counts().to_dict()
        st.markdown(
            f"- **Cluster {int(cluster_id)}** - {len(subset)} verbetes | "
            f"dominant quality: `{dominant}` | tipos: {tipo_counts} | "
            f"top terms: {', '.join(top_terms(frame, int(cluster_id)))}"
        )


def render_quality(frame: pd.DataFrame) -> None:
    """Render the quality control and anomaly tab."""
    st.subheader("Quality control and anomalies")
    st.caption(
        "Heuristic quality labels, IsolationForest anomaly scores and the "
        "documents flagged for manual review."
    )

    total = len(frame)
    n_clusters = frame.loc[frame["cluster_id"] != NOISE_CLUSTER, "cluster_id"].nunique()
    noise = int((frame["cluster_id"] == NOISE_CLUSTER).sum())
    anomaly_count = int((frame["anomaly_score"] > 0.7).sum())

    col_total, col_clusters, col_noise, col_anomaly = st.columns(4)
    col_total.metric("Documents", f"{total}")
    col_clusters.metric("Clusters", f"{n_clusters}")
    col_noise.metric("Noise (HDBSCAN)", f"{noise}")
    col_anomaly.metric(
        "Anomalies (> 0.7)",
        f"{anomaly_count}",
        help=f"{anomaly_count / total:.1%} of the corpus" if total else None,
    )

    col_quality, col_scores = st.columns(2)
    quality_counts = (
        frame["quality_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    figure_quality = px.bar(
        quality_counts,
        x="label",
        y="count",
        color="label",
        color_discrete_map=QUALITY_COLORS,
        title="Quality label distribution",
    )
    figure_quality.update_layout(showlegend=False)
    col_quality.plotly_chart(figure_quality, use_container_width=True)

    figure_scores = px.histogram(
        frame,
        x="anomaly_score",
        nbins=40,
        title="Anomaly score distribution (IsolationForest)",
    )
    col_scores.plotly_chart(figure_scores, use_container_width=True)

    figure_tipos = px.bar(
        frame["tipo"].value_counts().rename_axis("tipo").reset_index(name="count"),
        x="tipo",
        y="count",
        title="Documents by tipo",
    )
    figure_tipos.update_layout(showlegend=False)
    st.plotly_chart(figure_tipos, use_container_width=True)

    st.markdown("#### Flagged documents")
    threshold = st.slider(
        "Minimum anomaly score", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )
    flagged = frame[frame["anomaly_score"] >= threshold].sort_values(
        "anomaly_score", ascending=False
    )
    st.caption(f"{len(flagged)} documents at or above {threshold:.2f}")
    st.dataframe(
        flagged[
            [
                "doc_id",
                "tipo",
                "orgao",
                "quality_label",
                "cluster_id",
                "anomaly_score",
                "text",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Entry point of the Streamlit dashboard."""
    st.set_page_config(
        page_title="TST Jurisprudence Explorer",
        layout="wide",
    )

    st.title("TST Jurisprudence Explorer")
    st.caption(
        "Interactive exploration of the TST Livro de Jurisprudencia pipeline: "
        "Súmulas, Orientações Jurisprudenciais and Precedentes Normativos."
    )

    frame = load_data()
    if frame is None:
        return

    with st.sidebar:
        st.header("Dataset")
        st.metric("Verbetes", len(frame))
        st.write("By tipo", frame["tipo"].value_counts().to_dict())
        st.write("By quality", frame["quality_label"].value_counts().to_dict())
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.rerun()

    tab_search, tab_clusters, tab_quality = st.tabs(
        ["Search", "Cluster map", "Quality & anomalies"]
    )
    with tab_search:
        render_search(frame)
    with tab_clusters:
        render_clusters(frame)
    with tab_quality:
        render_quality(frame)


if __name__ == "__main__":
    main()
