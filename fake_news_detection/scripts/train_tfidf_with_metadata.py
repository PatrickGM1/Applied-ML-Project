import json
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from fake_news_detection.features.metadata import fit_metadata_transformers, transform_metadata

# TF‑IDF + metadata baseline.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLEANED_TEXT_DIR = PROJECT_DIR / 'data' / 'processed' / 'cleaned_text'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'baselines'
TEXT_COLUMN = 'statement_clean'

EXPERIMENTS = {
    'multiclass_text_metadata': {
        'train_file': 'train.processed.csv',
        'valid_file': 'valid.processed.csv',
        'label_column': 'label6_int',
    },
    'binary_text_metadata': {
        'train_file': 'train_binary.processed.csv',
        'valid_file': 'valid_binary.processed.csv',
        'label_column': 'label2_int',
    },
}


def load_dataframe(file_name):
    return pd.read_csv(CLEANED_TEXT_DIR / file_name)


def build_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def evaluate_split(name, train_df, valid_df, label_column):
    vectorizer = build_vectorizer()

    x_train_text = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(''))
    x_valid_text = vectorizer.transform(valid_df[TEXT_COLUMN].fillna(''))

    x_train_meta, meta_transformers = fit_metadata_transformers(train_df)
    x_valid_meta = transform_metadata(valid_df, meta_transformers)

    x_train = hstack([x_train_text, x_train_meta], format='csr')
    x_valid = hstack([x_valid_text, x_valid_meta], format='csr')

    y_train = train_df[label_column]
    y_valid = valid_df[label_column]

    classifier = LogisticRegression(max_iter=2000, solver='lbfgs')
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_valid)

    metrics = {
        'dataset': name,
        'train_rows': int(len(train_df)),
        'valid_rows': int(len(valid_df)),
        'label_column': label_column,
        'accuracy': float(accuracy_score(y_valid, predictions)),
        'f1_macro': float(f1_score(y_valid, predictions, average='macro')),
        'labels': sorted(pd.Series(y_valid).astype(str).unique().tolist()),
        'confusion_matrix': confusion_matrix(y_valid, predictions).tolist(),
        'classification_report': classification_report(y_valid, predictions, output_dict=True),
    }

    return metrics


def save_results(name, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f'{name}_metrics.json'
    txt_path = OUTPUT_DIR / f'{name}_summary.txt'

    with open(json_path, 'w', encoding='utf-8') as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_lines = [
        f"dataset: {metrics['dataset']}",
        f"label_column: {metrics['label_column']}",
        f"train_rows: {metrics['train_rows']}",
        f"valid_rows: {metrics['valid_rows']}",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"f1_macro: {metrics['f1_macro']:.4f}",
        f"labels: {', '.join(metrics['labels'])}",
        '',
        'confusion_matrix:',
        json.dumps(metrics['confusion_matrix']),
    ]

    with open(txt_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('\n'.join(summary_lines))


def main():
    results = {
        name: evaluate_split(
            name=name,
            train_df=load_dataframe(config['train_file']),
            valid_df=load_dataframe(config['valid_file']),
            label_column=config['label_column'],
        )
        for name, config in EXPERIMENTS.items()
    }

    for name, metrics in results.items():
        save_results(name, metrics)
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )


if __name__ == '__main__':
    main()
