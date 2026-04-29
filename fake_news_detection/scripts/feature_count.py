"""Compute total and active feature counts for TF-IDF + metadata pipelines.

This script mirrors the feature construction used in:
- fake_news_detection/scripts/train_tfidf_with_metadata.py
- fake_news_detection/scripts/final_text_metadata_test.py

It reports:
- Total feature dimension (number of columns)
- Active features (columns with at least one non-zero value in the train matrix)
- A breakdown of feature groups (TF-IDF, subjects, categoricals, history)

Run (from repo root):
  python -m fake_news_detection.scripts.feature_count

Or on Windows with the venv interpreter:
  .venv\\Scripts\\python.exe -m fake_news_detection.scripts.feature_count
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from fake_news_detection.features.metadata import (
    HISTORY_COLUMNS,
    fit_metadata_transformers,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"
TEXT_COLUMN = "statement_clean"


@dataclass(frozen=True)
class FeatureCounts:
    tfidf_features: int
    subjects_features: int
    categorical_features: int
    history_features: int

    @property
    def total(self) -> int:
        return (
            self.tfidf_features
            + self.subjects_features
            + self.categorical_features
            + self.history_features
        )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def load_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(CLEANED_TEXT_DIR / file_name)


def compute_counts(train_df: pd.DataFrame) -> tuple[FeatureCounts, int, int]:
    vectorizer = build_vectorizer()
    x_text = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(""))

    x_meta, meta_transformers = fit_metadata_transformers(train_df)
    x_all = hstack([x_text, x_meta], format="csr")

    # Metadata breakdown sizes
    subjects_features = len(meta_transformers.subjects_encoder.classes_)
    categorical_features = int(meta_transformers.categorical_encoder.get_feature_names_out().shape[0])
    history_features = len(HISTORY_COLUMNS)

    counts = FeatureCounts(
        tfidf_features=int(x_text.shape[1]),
        subjects_features=int(subjects_features),
        categorical_features=int(categorical_features),
        history_features=int(history_features),
    )

    total_features = int(x_all.shape[1])
    active_features = int((x_all.getnnz(axis=0) > 0).sum())

    # Sanity check: derived total should match matrix width.
    if counts.total != total_features:
        raise RuntimeError(
            f"Feature breakdown mismatch: breakdown_total={counts.total} != matrix_total={total_features}"
        )

    return counts, total_features, active_features


def main() -> None:
    experiments = {
        "multiclass_train": "train.processed.csv",
        "binary_train": "train_binary.processed.csv",
    }

    for name, train_file in experiments.items():
        train_df = load_dataframe(train_file)
        counts, total_features, active_features = compute_counts(train_df)

        print(f"\n== {name} ==")
        print(f"train_rows: {len(train_df)}")
        print(f"tfidf_features: {counts.tfidf_features}")
        print(f"subjects_features: {counts.subjects_features}")
        print(f"categorical_features: {counts.categorical_features}")
        print(f"history_features: {counts.history_features}")
        print(f"TOTAL features: {total_features}")
        print(f"ACTIVE features (non-zero cols in train): {active_features}")


if __name__ == "__main__":
    main()
