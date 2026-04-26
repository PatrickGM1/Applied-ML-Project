import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
LABELED_DIR = PROJECT_DIR / 'data' / 'processed' / 'labeled'
ENCODER_DIR = PROJECT_DIR / 'artifacts' / 'encoders'

RAW_COLUMNS = [
    'id',
    'label',
    'statement',
    'subjects',
    'speaker',
    'speaker_job',
    'state',
    'party',
    'hist1',
    'hist2',
    'hist3',
    'hist4',
    'hist5',
    'context',
]

BINARY_LABEL_MAP = {
    'pants-fire': 'fake',
    'false': 'fake',
    'mostly-true': 'real',
    'true': 'real',
}


def load_split(file_name):
    return pd.read_csv(
        RAW_DIR / file_name,
        sep='\t',
        header=None,
        names=RAW_COLUMNS,
        engine='python',
        quoting=3,
        dtype=str,
    )


def save_pickle(path, obj):
    with open(path, 'wb') as file_handle:
        pickle.dump(obj, file_handle)


def add_binary_label_column(frame):
    labeled = frame.copy()
    labeled['label_binary'] = labeled['label'].map(BINARY_LABEL_MAP)
    return labeled


def build_binary_subset(frame):
    return frame[frame['label_binary'].notna()].copy()


def encode_labels(train_frame, other_frames, source_column, target_column):
    encoder = LabelEncoder().fit(train_frame[source_column])
    train_frame[target_column] = encoder.transform(train_frame[source_column])

    for frame in other_frames:
        frame[target_column] = encoder.transform(frame[source_column])

    return encoder


def save_processed_splits(splits, suffix=''):
    for split_name, frame in splits.items():
        file_name = f'{split_name}{suffix}.processed.csv'
        frame.to_csv(LABELED_DIR / file_name, index=False)


def main():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    multiclass_splits = {
        'train': add_binary_label_column(load_split('train.tsv')),
        'valid': add_binary_label_column(load_split('valid.tsv')),
        'test': add_binary_label_column(load_split('test.tsv')),
    }

    binary_splits = {
        split_name: build_binary_subset(frame)
        for split_name, frame in multiclass_splits.items()
    }

    label_encoder_6 = encode_labels(
        multiclass_splits['train'],
        [multiclass_splits['valid'], multiclass_splits['test']],
        'label',
        'label6_int',
    )
    label_encoder_2 = encode_labels(
        binary_splits['train'],
        [binary_splits['valid'], binary_splits['test']],
        'label_binary',
        'label2_int',
    )

    save_pickle(ENCODER_DIR / 'label_encoder_6.pkl', label_encoder_6)
    save_pickle(ENCODER_DIR / 'label_encoder_2.pkl', label_encoder_2)

    save_processed_splits(multiclass_splits)
    save_processed_splits(binary_splits, suffix='_binary')


if __name__ == '__main__':
    main()