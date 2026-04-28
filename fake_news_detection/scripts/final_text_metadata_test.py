import json
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from fake_news_detection.features.metadata import fit_metadata_transformers, transform_metadata

# Final evaluation: train on train+valid, test once on test.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "final"
TEXT_COLUMN = "statement_clean"
SUBJECTS_COLUMN = "subjects"

EXPERIMENTS = {
    "multiclass_text_metadata_final": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file": "test.processed.csv",
        "label_column": "label6_int",
    },
    "binary_text_metadata_final": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file": "test_binary.processed.csv",
        "label_column": "label2_int",
    },
}


def load_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(CLEANED_TEXT_DIR / file_name)


def load_and_concat(files: list[str]) -> pd.DataFrame:
    frames = [load_dataframe(file_name) for file_name in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, solver="lbfgs")


def compute_metrics(name: str, y_true, y_pred) -> dict:
    return {
        "dataset": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "labels": sorted(pd.Series(y_true).astype(str).unique().tolist()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def extract_subjects(frame: pd.DataFrame) -> set[str]:
    subjects = set()
    for value in frame[SUBJECTS_COLUMN].dropna():
        for token in str(value).split(","):
            token = token.strip().lower()
            if token:
                subjects.add(token)
    return subjects


def save_results(
    name: str,
    metrics: dict,
    train_rows: int,
    test_rows: int,
    label_column: str,
    unknown_subjects: list[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        **metrics,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "label_column": label_column,
        "unknown_subjects": unknown_subjects,
    }

    json_path = OUTPUT_DIR / f"{name}_metrics.json"
    txt_path = OUTPUT_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_lines = [
        f"dataset: {metrics['dataset']}",
        f"label_column: {metrics['label_column']}",
        f"train_rows: {metrics['train_rows']}",
        f"test_rows: {metrics['test_rows']}",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"f1_macro: {metrics['f1_macro']:.4f}",
        f"labels: {', '.join(metrics['labels'])}",
        f"unknown_subjects: {', '.join(unknown_subjects) if unknown_subjects else 'none'}",
        "",
        "confusion_matrix:",
        json.dumps(metrics["confusion_matrix"]),
    ]

    with open(txt_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))


def evaluate_text_metadata(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str) -> dict:
    vectorizer = build_vectorizer()
    x_train_text = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(""))
    x_test_text = vectorizer.transform(test_df[TEXT_COLUMN].fillna(""))

    x_train_meta, transformers = fit_metadata_transformers(train_df)
    x_test_meta = transform_metadata(test_df, transformers)

    x_train = hstack([x_train_text, x_train_meta], format="csr")
    x_test = hstack([x_test_text, x_test_meta], format="csr")

    classifier = build_classifier()
    classifier.fit(x_train, train_df[label_column])
    predictions = classifier.predict(x_test)

    metrics = compute_metrics(name, test_df[label_column], predictions)
    # Logs info useful for report for unknown classes
    known_subjects = set(transformers.subjects_encoder.classes_)
    test_subjects = extract_subjects(test_df)
    unknown_subjects = sorted(test_subjects - known_subjects)
    save_results(name, metrics, len(train_df), len(test_df), label_column, unknown_subjects)
    return metrics


def main() -> None:
    results = {}
    for name, config in EXPERIMENTS.items():
        train_df = load_and_concat(config["train_files"])
        test_df = load_dataframe(config["test_file"])
        label_column = config["label_column"]

        results[name] = evaluate_text_metadata(name, train_df, test_df, label_column)

    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
