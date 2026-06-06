"""Compare BERT vs TF-IDF baseline — no history features.
----------------------------
The LIAR dataset includes speaker history counts (hist1–hist5) that encode
past credibility. This script compares BERT and TF-IDF models WITHOUT those
features so the comparison is fair and not inflated by history leakage.

BERT text-only results are loaded from previously saved JSON artifacts
(no retraining needed). TF-IDF baselines are trained from scratch (fast).

Three models per task (binary + multiclass):
    1. TF-IDF text-only            (baseline, no metadata at all)
    2. TF-IDF text + meta (no hist)(baseline with subjects + categoricals)
    3. BERT text-only              (loaded from saved artifacts)

Outputs:
    artifacts/comparisons_no_history/<name>_metrics.json
    artifacts/comparisons_no_history/<name>_summary.txt
    artifacts/comparisons_no_history/comparison_table.txt

Run:
    python fake_news_detection/scripts/compare_bert_vs_baseline_no_history.py
"""

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "comparisons_no_history"

BERT_TEXT_ONLY_DIR = PROJECT_DIR / "artifacts" / "final" / "bert_text_only"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEXT_COLUMN = "statement_clean"
SUBJECTS_COL = "subjects"
CATEGORICAL_COLS = ["party", "state", "speaker_job"]
# hist1–hist5 deliberately excluded

EXPERIMENTS = {
    "multiclass": {
        "train_file": "train.processed.csv",
        "test_file": "test.processed.csv",
        "label_column": "label6_int",
        "bert_json": "multiclass_bert_text_only_metrics.json",
    },
    "binary": {
        "train_file": "train_binary.processed.csv",
        "test_file": "test_binary.processed.csv",
        "label_column": "label2_int",
        "bert_json": "binary_bert_text_only_metrics.json",
    },
}


# ---------------------------------------------------------------------------
# Metadata preprocessing (no history)
# ---------------------------------------------------------------------------

def _parse_subjects(df: pd.DataFrame) -> list[list[str]]:
    result = []
    for val in df[SUBJECTS_COL].fillna(""):
        tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
        result.append(tokens)
    return result


class MetadataNoHistorySparse:
    """Sparse metadata pipeline: subjects multi-hot + categoricals one-hot. No history."""

    def __init__(self):
        self.subjects_encoder = MultiLabelBinarizer(sparse_output=True)
        self.categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    def fit(self, df: pd.DataFrame) -> csr_matrix:
        subjects = _parse_subjects(df)
        subjects_matrix = self.subjects_encoder.fit_transform(subjects)
        cat_df = df[CATEGORICAL_COLS].fillna("missing").astype(str)
        categorical_matrix = self.categorical_encoder.fit_transform(cat_df)
        return hstack([subjects_matrix, categorical_matrix], format="csr")

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        subjects = _parse_subjects(df)
        subjects_matrix = self.subjects_encoder.transform(subjects)
        cat_df = df[CATEGORICAL_COLS].fillna("missing").astype(str)
        categorical_matrix = self.categorical_encoder.transform(cat_df)
        return hstack([subjects_matrix, categorical_matrix], format="csr")


# ---------------------------------------------------------------------------
# TF-IDF models
# ---------------------------------------------------------------------------

def _build_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def run_tfidf_text_only(train_df, test_df, label_column, name):
    vectorizer = _build_vectorizer()
    x_train = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(""))
    x_test = vectorizer.transform(test_df[TEXT_COLUMN].fillna(""))

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(x_train, train_df[label_column])
    preds = clf.predict(x_test)

    return _compute_metrics(name, test_df[label_column].values, preds, len(train_df), len(test_df))


def run_tfidf_text_meta_no_hist(train_df, test_df, label_column, name):
    vectorizer = _build_vectorizer()
    x_train_text = vectorizer.fit_transform(train_df[TEXT_COLUMN].fillna(""))
    x_test_text = vectorizer.transform(test_df[TEXT_COLUMN].fillna(""))

    meta = MetadataNoHistorySparse()
    x_train_meta = meta.fit(train_df)
    x_test_meta = meta.transform(test_df)

    x_train = hstack([x_train_text, x_train_meta], format="csr")
    x_test = hstack([x_test_text, x_test_meta], format="csr")

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(x_train, train_df[label_column])
    preds = clf.predict(x_test)

    return _compute_metrics(name, test_df[label_column].values, preds, len(train_df), len(test_df))


# ---------------------------------------------------------------------------
# Load saved BERT results
# ---------------------------------------------------------------------------

def load_bert_metrics(json_filename, display_name):
    json_path = BERT_TEXT_ONLY_DIR / json_filename
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    return {
        "dataset": display_name,
        "train_rows": raw["train_rows"],
        "eval_rows": raw["eval_rows"],
        "accuracy": raw["accuracy"],
        "f1_macro": raw["f1_macro"],
        "f1_weighted": raw["f1_weighted"],
        "labels": raw["labels"],
        "confusion_matrix": raw["confusion_matrix"],
        "classification_report": raw["classification_report"],
    }


# ---------------------------------------------------------------------------
# Metrics / IO helpers
# ---------------------------------------------------------------------------

def _compute_metrics(name, y_true, y_pred, train_rows, eval_rows):
    return {
        "dataset": name,
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": sorted(pd.Series(y_true).astype(str).unique().tolist()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def _save_results(name, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{name}_metrics.json"
    txt_path = OUTPUT_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    summary_lines = [
        f"dataset: {metrics['dataset']}",
        f"train_rows: {metrics['train_rows']}",
        f"eval_rows: {metrics['eval_rows']}",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"f1_macro: {metrics['f1_macro']:.4f}",
        f"f1_weighted: {metrics['f1_weighted']:.4f}",
        f"labels: {', '.join(metrics['labels'])}",
        "",
        "confusion_matrix:",
        json.dumps(metrics["confusion_matrix"]),
    ]
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))


def _print_comparison_table(all_results: dict[str, dict]):
    header = f"{'Model':<45} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}"
    sep = "-" * len(header)

    lines = [sep, header, sep]

    prev_task = None
    for name, m in all_results.items():
        task = name.split("_")[0]
        if prev_task and task != prev_task:
            lines.append(sep)
        prev_task = task
        lines.append(
            f"{name:<45} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['f1_weighted']:>12.4f}"
        )
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "comparison_table.txt", "w", encoding="utf-8") as fh:
        fh.write(table + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data  : {CLEANED_TEXT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    all_results: dict[str, dict] = {}

    for task, config in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"  TASK: {task}")
        print(f"{'='*70}")

        train_df = pd.read_csv(CLEANED_TEXT_DIR / config["train_file"])
        test_df = pd.read_csv(CLEANED_TEXT_DIR / config["test_file"])
        label_column = config["label_column"]

        print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

        # 1. TF-IDF text-only
        print(f"\n--- {task}: TF-IDF text-only ---")
        name = f"{task}_tfidf_text_only"
        m = run_tfidf_text_only(train_df, test_df, label_column, name)
        _save_results(name, m)
        all_results[name] = m
        print(f"  accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")

        # 2. TF-IDF text + metadata (no history)
        print(f"\n--- {task}: TF-IDF text + meta (no history) ---")
        name = f"{task}_tfidf_meta_no_hist"
        m = run_tfidf_text_meta_no_hist(train_df, test_df, label_column, name)
        _save_results(name, m)
        all_results[name] = m
        print(f"  accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")

        # 3. BERT text-only (loaded from saved JSON — no history by design)
        print(f"\n--- {task}: BERT text-only (loaded from saved results) ---")
        name = f"{task}_bert_text_only"
        m = load_bert_metrics(config["bert_json"], name)
        _save_results(name, m)
        all_results[name] = m
        print(f"  accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")

    # Final comparison table
    _print_comparison_table(all_results)
    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
