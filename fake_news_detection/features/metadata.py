from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

# metadata pipeline: subjects multi-hot; party/state/speaker_job one-hot; 
# hist1–hist5 numeric + scaling with explicit missing handling.

SUBJECTS_COLUMN = 'subjects'
CATEGORICAL_COLUMNS = ['party', 'state', 'speaker_job']
HISTORY_COLUMNS = ['hist1', 'hist2', 'hist3', 'hist4', 'hist5']


@dataclass
class MetadataTransformers:
    subjects_encoder: MultiLabelBinarizer
    categorical_encoder: OneHotEncoder
    history_scaler: StandardScaler


def _split_subjects(value: str | float | None) -> List[str]:
    if value is None or pd.isna(value):
        return []
    subjects = [token.strip().lower() for token in str(value).split(',')]
    return [token for token in subjects if token]


def _prepare_subjects(series: pd.Series) -> List[List[str]]:
    return [_split_subjects(value) for value in series]


def _prepare_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    # Missing values are mapped to a single explicit category.
    return frame[CATEGORICAL_COLUMNS].fillna('missing').astype(str)


def _prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    history = frame[HISTORY_COLUMNS].apply(pd.to_numeric, errors='coerce')
    # Missing counts are treated as 0 for history features.
    return history.fillna(0.0)


def fit_metadata_transformers(train_frame: pd.DataFrame) -> tuple[csr_matrix, MetadataTransformers]:
    subjects = _prepare_subjects(train_frame[SUBJECTS_COLUMN])
    subjects_encoder = MultiLabelBinarizer(sparse_output=True)
    subjects_matrix = subjects_encoder.fit_transform(subjects)

    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    categorical_matrix = categorical_encoder.fit_transform(_prepare_categoricals(train_frame))

    history_scaler = StandardScaler()
    history_values = _prepare_history(train_frame)
    history_matrix = csr_matrix(history_scaler.fit_transform(history_values))

    combined = hstack([subjects_matrix, categorical_matrix, history_matrix], format='csr')

    transformers = MetadataTransformers(
        subjects_encoder=subjects_encoder,
        categorical_encoder=categorical_encoder,
        history_scaler=history_scaler,
    )
    return combined, transformers


def transform_metadata(frame: pd.DataFrame, transformers: MetadataTransformers) -> csr_matrix:
    subjects = _prepare_subjects(frame[SUBJECTS_COLUMN])
    subjects_matrix = transformers.subjects_encoder.transform(subjects)

    categorical_matrix = transformers.categorical_encoder.transform(_prepare_categoricals(frame))

    history_values = _prepare_history(frame)
    history_matrix = csr_matrix(transformers.history_scaler.transform(history_values))

    return hstack([subjects_matrix, categorical_matrix, history_matrix], format='csr')


