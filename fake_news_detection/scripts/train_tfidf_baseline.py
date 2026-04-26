import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLEANED_TEXT_DIR = PROJECT_DIR / 'data' / 'processed' / 'cleaned_text'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / 'baselines'
TEXT_COLUMN = 'statement_clean'

EXPERIMENTS = {
    'multiclass': {
        'train_file': 'train.processed.csv',
        'valid_file': 'valid.processed.csv',
        'label_column': 'label6_int',
    },
    'binary': {
        'train_file': 'train_binary.processed.csv',
        'valid_file': 'valid_binary.processed.csv',
        'label_column': 'label2_int',
    },
}


def load_dataframe(file_name):
    return pd.read_csv(CLEANED_TEXT_DIR / file_name)


def build_pipeline():
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
    )
    return vectorizer, classifier


def evaluate_split(name, train_df, valid_df, label_column):
    vectorizer, classifier = build_pipeline()

    x_train = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(''))
    x_valid = vectorizer.transform(valid_df[TEXT_COLUMN].fillna(''))
    y_train = train_df[label_column]
    y_valid = valid_df[label_column]

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


def run_experiment(name, config):
    train_df = load_dataframe(config['train_file'])
    valid_df = load_dataframe(config['valid_file'])

    metrics = evaluate_split(
        name=name,
        train_df=train_df,
        valid_df=valid_df,
        label_column=config['label_column'],
    )
    save_results(name, metrics)
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
        name: run_experiment(name, config)
        for name, config in EXPERIMENTS.items()
    }

    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )


if __name__ == '__main__':
    main()