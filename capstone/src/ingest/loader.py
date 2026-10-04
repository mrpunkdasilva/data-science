"""Data ingestion from HuggingFace datasets."""

import pandas as pd
from pathlib import Path
from datasets import load_dataset
from config import settings


def load_reviews(
    categories: list[str] | None = None,
    limit: int = 15000,
    min_text_len: int = 50,
    max_text_len: int = 500,
    verified_only: bool = True,
) -> pd.DataFrame:
    """
    Load reviews from HuggingFace Amazon Reviews 2023 dataset.

    Args:
        categories: List of HF dataset subset names (e.g., "raw_review_Electronics")
        limit: Total number of reviews to load across all categories
        min_text_len: Minimum review text length
        max_text_len: Maximum review text length
        verified_only: Only include verified purchases

    Returns:
        DataFrame with review data and derived features
    """
    if categories is None:
        categories = settings.CATEGORIES

    # Ensure output directory exists
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    per_category = limit // len(categories)
    all_dfs = []

    for cat in categories:
        console_log(f"Loading {cat} (target: {per_category})...")
        ds = load_dataset(
            settings.HF_DATASET, cat, split="full", streaming=True, trust_remote_code=True
        )

        rows = []
        for row in ds:
            text = row.get("text", "") or ""
            title = row.get("title", "") or ""

            if len(text) < min_text_len or len(text) > max_text_len:
                continue
            if verified_only and not row.get("verified_purchase", False):
                continue

            rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "asin": row.get("asin", ""),
                    "user_id": row.get("user_id", ""),
                    "rating": int(row.get("rating", 0)),
                    "title": title[:200],
                    "text": text[:2000],
                    "helpful_vote": int(row.get("helpful_vote", 0)),
                    "verified_purchase": bool(row.get("verified_purchase", False)),
                    "timestamp": int(row.get("timestamp", 0)),
                    "category": cat.replace("raw_review_", ""),
                }
            )

            if len(rows) >= per_category:
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            df["text_len"] = df["text"].str.len()
            df["quality_label"] = df.apply(_heuristic_quality_label, axis=1)
            all_dfs.append(df)
            console_log(f"  Loaded {len(df)} reviews from {cat}")

    result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    console_log(f"Total loaded: {len(result)} reviews")
    return result


def _heuristic_quality_label(row: pd.Series) -> str:
    """Assign heuristic quality label based on metadata."""
    text = (row.get("text", "") or "").lower()
    helpful = row.get("helpful_vote", 0)
    rating = row.get("rating", 0)
    text_len = row.get("text_len", 0)

    if helpful >= 10 and rating >= 4 and text_len > 100:
        return "helpful"
    if helpful >= 5 and any(kw in text for kw in ["funny", "lol", "😂", "haha", "hilarious"]):
        return "funny"
    if (
        rating <= 2
        and helpful >= 5
        and any(kw in text for kw in ["weird", "strange", "bizarre", "odd", "unusual"])
    ):
        return "weird"
    if (
        rating == 5
        and helpful == 0
        and text_len < 80
        and any(
            kw in text
            for kw in ["best ever", "perfect", "amazing product", "highly recommend", "five stars"]
        )
    ):
        return "fake_suspect"
    return "normal"


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[INGEST] {msg}")
