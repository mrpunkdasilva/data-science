"""Quality classification using heuristic labels + supervised learning."""

# mypy: ignore-errors

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import settings
from store.client import get_chroma_client, get_collection
from chromadb.api.models.Collection import Collection


def run_quality_classification(collection_name: str) -> None:
    """
    Train classifier on heuristic labels, predict for all reviews.

    Updates ChromaDB metadata with: quality_label (predicted)
    """
    client = get_chroma_client()
    collection = get_collection(collection_name)  # type: ignore[assignment]

    console_log("Fetching data for quality classification...")
    count = collection.count()
    result = collection.get(include=["documents", "metadatas"], limit=count)  # type: ignore[arg-type]

    texts = result["documents"] or []
    metadatas = result["metadatas"] or []
    ids = result["ids"] or []

    # Extract heuristic labels
    y_heuristic = [m.get("quality_label", "normal") for m in metadatas]
    unique_labels = set(y_heuristic)
    console_log(f"Heuristic label distribution: {pd.Series(y_heuristic).value_counts().to_dict()}")

    # Need at least 2 classes
    if len(unique_labels) < 2:
        console_log("Only one class found, skipping supervised training")
        return

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=settings.CLASSIFIER_MAX_FEATURES,
        ngram_range=settings.CLASSIFIER_NGRAM_RANGE,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)

    # Check class distribution for stratify
    from collections import Counter

    label_counts = Counter(y_heuristic)
    min_count = min(label_counts.values())
    use_stratify = min_count >= 2

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_heuristic,
        test_size=settings.CLASSIFIER_TEST_SIZE,
        random_state=settings.CLASSIFIER_RANDOM_STATE,
        stratify=y_heuristic if use_stratify else None,
    )

    if not use_stratify:
        console_log("Some classes have <2 samples, disabling stratify")

    # Train LogisticRegression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=settings.CLASSIFIER_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    console_log(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Predict all
    y_all_pred = clf.predict(X)

    # Save model
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": clf},
        settings.MODELS_DIR / "quality_classifier.joblib",
    )

    # Update ChromaDB
    console_log("Updating ChromaDB with predicted quality labels...")
    new_metadatas = []
    for m, pred in zip(metadatas, y_all_pred):
        m = dict(m)
        m["quality_label"] = pred
        new_metadatas.append(m)

    collection.update(ids=ids, metadatas=new_metadatas)  # type: ignore[arg-type]
    console_log("Quality classification complete!")


def console_log(msg: str) -> None:
    """Simple console logging."""
    print(f"[QUALITY] {msg}")
