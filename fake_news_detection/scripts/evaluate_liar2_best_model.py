import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fake_news_detection.scripts.bert_text_metadata import (
    BertMetadataFusion,
    FusionDataset,
    MetadataTransformers,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

LIAR2_CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "liar2" / "processed" / "cleaned_text"
LIAR2_TEST_PATH = LIAR2_CLEANED_TEXT_DIR / "test_binary.processed.csv"

ORIGINAL_LIAR_RAW_DIR = PROJECT_DIR / "data" / "raw"

MODEL_DIR = PROJECT_DIR / "artifacts" / "models" / "bert_text_metadata" / "binary_bert_metadata"
MODEL_WEIGHTS_PATH = MODEL_DIR / "model_weights.pt"
META_TRANSFORMERS_PATH = MODEL_DIR / "meta_transformers.pkl"
SERVING_CONFIG_PATH = MODEL_DIR / "serving_config.json"

OUTPUT_DIR = PROJECT_DIR / "artifacts" / "liar2"


# ---------------------------------------------------------------------------
# Project label logic
# ---------------------------------------------------------------------------

# This follows your actual create_labels.py logic, not the README wording.
# Your binary BERT model was trained only on:
# fake = false + pants-fire
# real = mostly-true + true
# It did not train on barely-true or half-true.
PROJECT_BINARY_LABEL_MAP = {
    "pants-fire": "fake",
    "false": "fake",
    "mostly-true": "real",
    "true": "real",
}

PROJECT_BINARY_TO_INT = {
    "fake": 0,
    "real": 1,
}

RAW_LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "hist1",
    "hist2",
    "hist3",
    "hist4",
    "hist5",
    "context",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_statement_for_matching(text) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_column(df: pd.DataFrame, column_name: str, default_value):
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default_value] * len(df), index=df.index)


def load_original_liar_for_label_inference() -> pd.DataFrame:
    frames = []

    for file_name in ["train.tsv", "valid.tsv", "test.tsv"]:
        file_path = ORIGINAL_LIAR_RAW_DIR / file_name

        if not file_path.exists():
            continue

        frame = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=RAW_LIAR_COLUMNS,
            engine="python",
            quoting=3,
            dtype=str,
        )

        frames.append(frame[["statement", "label"]])

    if not frames:
        raise FileNotFoundError(
            "Could not find original LIAR train.tsv, valid.tsv, or test.tsv. "
            "These files are needed to infer the LIAR2 numeric label mapping."
        )

    original = pd.concat(frames, ignore_index=True)
    original["statement_key"] = original["statement"].map(normalize_statement_for_matching)
    original = original.drop_duplicates(subset=["statement_key"])

    return original


def infer_liar2_numeric_label_map(liar2_frame: pd.DataFrame) -> dict[int, str]:
    """
    LIAR2 may store labels as numbers instead of label strings.

    To avoid guessing the numeric label meaning, this function matches overlapping
    statements between original LIAR and LIAR2, then infers which numeric LIAR2
    label corresponds to which original LIAR label.
    """
    original = load_original_liar_for_label_inference()

    temp = liar2_frame[["statement", "label"]].copy()
    temp["statement_key"] = temp["statement"].map(normalize_statement_for_matching)

    merged = temp.merge(
        original[["statement_key", "label"]],
        on="statement_key",
        how="inner",
        suffixes=("_liar2", "_original"),
    )

    if merged.empty:
        raise ValueError(
            "Could not infer LIAR2 numeric labels because no overlapping statements "
            "were found between LIAR2 and original LIAR."
        )

    label_map = {}

    print("\nInferred LIAR2 numeric label mapping:")

    for numeric_label, group in merged.groupby("label_liar2"):
        counts = Counter(group["label_original"])
        best_label, best_count = counts.most_common(1)[0]

        label_map[int(numeric_label)] = best_label

        agreement = best_count / len(group)
        print(
            f"  {int(numeric_label)} -> {best_label} "
            f"based on {best_count}/{len(group)} overlapping rows "
            f"(agreement={agreement:.3f})"
        )

    print()

    return label_map


def convert_liar2_labels_to_names(liar2_frame: pd.DataFrame) -> pd.Series:
    raw_labels = liar2_frame["label"]

    # Case 1: labels are already text labels.
    if raw_labels.dtype == object:
        normalized = (
            raw_labels
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("barely true", "barely-true", regex=False)
            .str.replace("half true", "half-true", regex=False)
            .str.replace("mostly true", "mostly-true", regex=False)
            .str.replace("pants on fire", "pants-fire", regex=False)
            .str.replace("mostly false", "barely-true", regex=False)
            .str.replace("mostly-false", "barely-true", regex=False)
        )

        known_labels = {
            "pants-fire",
            "false",
            "barely-true",
            "half-true",
            "mostly-true",
            "true",
        }

        if set(normalized.dropna().unique()).issubset(known_labels):
            return normalized

    # Case 2: labels are numeric.
    numeric_labels = pd.to_numeric(raw_labels, errors="raise").astype(int)
    inferred_map = infer_liar2_numeric_label_map(liar2_frame)

    label_names = numeric_labels.map(inferred_map)

    if label_names.isna().any():
        unknown = sorted(numeric_labels[label_names.isna()].unique().tolist())
        raise ValueError(f"Could not map these LIAR2 numeric labels: {unknown}")

    return label_names


def convert_liar2_to_project_schema(liar2_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Converts LIAR2 columns to the same schema expected by your BERT + metadata model.
    """
    frame = pd.DataFrame(index=liar2_frame.index)

    frame["id"] = get_column(liar2_frame, "id", "").astype(str)

    frame["label_name"] = convert_liar2_labels_to_names(liar2_frame)

    frame["statement"] = get_column(liar2_frame, "statement", "").fillna("").astype(str)

    # LIAR2 commonly uses "subject"; your project uses "subjects".
    # Your project separates multiple subjects with commas, while LIAR2 may use semicolons.
    frame["subjects"] = (
        get_column(liar2_frame, "subject", "")
        .fillna("")
        .astype(str)
        .str.replace(";", ",", regex=False)
    )

    frame["speaker"] = get_column(liar2_frame, "speaker", "").fillna("missing").astype(str)

    # LIAR2 has speaker_description rather than speaker_job.
    frame["speaker_job"] = (
        get_column(liar2_frame, "speaker_description", "missing")
        .fillna("missing")
        .astype(str)
    )

    frame["state"] = (
        get_column(liar2_frame, "state_info", "missing")
        .fillna("missing")
        .astype(str)
    )

    # LIAR2 does not provide party exactly like original LIAR.
    frame["party"] = "missing"

    # Original project history columns:
    # hist1 = barely-true / mostly-false count
    # hist2 = false count
    # hist3 = half-true count
    # hist4 = mostly-true count
    # hist5 = pants-fire count
    frame["hist1"] = pd.to_numeric(
        get_column(liar2_frame, "mostly_false_counts", 0),
        errors="coerce",
    ).fillna(0)

    frame["hist2"] = pd.to_numeric(
        get_column(liar2_frame, "false_counts", 0),
        errors="coerce",
    ).fillna(0)

    frame["hist3"] = pd.to_numeric(
        get_column(liar2_frame, "half_true_counts", 0),
        errors="coerce",
    ).fillna(0)

    frame["hist4"] = pd.to_numeric(
        get_column(liar2_frame, "mostly_true_counts", 0),
        errors="coerce",
    ).fillna(0)

    frame["hist5"] = pd.to_numeric(
        get_column(liar2_frame, "pants_on_fire_counts", 0),
        errors="coerce",
    ).fillna(0)

    frame["context"] = get_column(liar2_frame, "context", "").fillna("").astype(str)

    frame["label_binary"] = frame["label_name"].map(PROJECT_BINARY_LABEL_MAP)
    frame["label2_int"] = frame["label_binary"].map(PROJECT_BINARY_TO_INT)

    return frame


def load_meta_transformers():
    """
    Handles both possible pickle situations:

    1. The model was trained with:
       python -m fake_news_detection.scripts.bert_text_metadata

    2. The model was trained with:
       python fake_news_detection/scripts/bert_text_metadata.py

    In the second case, pickle may remember MetadataTransformers as __main__.
    This small compatibility line prevents loading errors.
    """
    setattr(sys.modules["__main__"], "MetadataTransformers", MetadataTransformers)

    with open(META_TRANSFORMERS_PATH, "rb") as file_handle:
        return pickle.load(file_handle)


def evaluate_binary_bert_metadata_on_liar2() -> dict:
    if not LIAR2_TEST_PATH.exists():
        raise FileNotFoundError(
            f"LIAR2 processed binary test file not found: {LIAR2_TEST_PATH}\n"
            "First run:\n"
            "python -m fake_news_detection.scripts.prepare_liar2_data"
        )

    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"BERT weights not found: {MODEL_WEIGHTS_PATH}\n"
            "First run:\n"
            "python -m fake_news_detection.scripts.bert_text_metadata"
        )

    if not META_TRANSFORMERS_PATH.exists():
        raise FileNotFoundError(
            f"Metadata transformers not found: {META_TRANSFORMERS_PATH}\n"
            "First run:\n"
            "python -m fake_news_detection.scripts.bert_text_metadata"
        )

    if not SERVING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Serving config not found: {SERVING_CONFIG_PATH}\n"
            "First run:\n"
            "python -m fake_news_detection.scripts.bert_text_metadata"
        )

    print(f"Reading processed LIAR2 binary test set from: {LIAR2_TEST_PATH}")
    liar2_project = pd.read_csv(LIAR2_TEST_PATH)
    liar2_project.columns = [column.strip() for column in liar2_project.columns]

    print(f"Processed LIAR2 binary rows: {len(liar2_project)}")

    required_columns = {
        "id",
        "label",
        "statement",
        "subjects",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "hist1",
        "hist2",
        "hist3",
        "hist4",
        "hist5",
        "context",
        "label_binary",
        "label2_int",
    }

    missing_columns = required_columns - set(liar2_project.columns)
    if missing_columns:
        raise ValueError(
            "The processed LIAR2 binary test file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # test_binary.processed.csv already excludes barely-true and half-true.
    # This filter is kept only as a safety check.
    usable = liar2_project[liar2_project["label2_int"].notna()].copy()
    usable["label2_int"] = usable["label2_int"].astype(int)

    # The previous raw-conversion script used label_name.
    # The processed LIAR2 file uses label, so this alias keeps the output format unchanged.
    if "label_name" not in usable.columns:
        usable["label_name"] = usable["label"].astype(str)

    skipped = len(liar2_project) - len(usable)

    print(f"Rows used for binary evaluation: {len(usable)}")
    print(f"Rows skipped because label2_int is missing: {skipped}")
    print("\nBinary label distribution:")
    print(usable["label_binary"].value_counts(dropna=False))

    with open(SERVING_CONFIG_PATH, "r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    model_name = config["model_name"]
    num_labels = int(config["num_labels"])
    meta_dim = int(config["meta_dim"])
    max_length = int(config["max_length"])

    if num_labels != 2:
        raise ValueError(
            f"Expected binary model with num_labels=2, but found num_labels={num_labels}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    meta_transformers = load_meta_transformers()

    metadata_matrix = meta_transformers.transform(usable)

    dataset = FusionDataset(
        df=usable,
        tokenizer=tokenizer,
        metadata_matrix=metadata_matrix,
        text_column="statement",
        label_column="label2_int",
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertMetadataFusion(
        bert_model_name=model_name,
        meta_dim=meta_dim,
        num_labels=num_labels,
    )

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                metadata=metadata,
            )

            predictions = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    metrics = {
        "dataset": "liar2",
        "split": "test",
        "model": "binary_bert_metadata",
        "model_weights_path": str(MODEL_WEIGHTS_PATH),
        "liar2_test_path": str(LIAR2_TEST_PATH),
        "rows_raw": int(len(liar2_project)),
        "rows_evaluated": int(len(usable)),
        "rows_skipped": int(skipped),
        "skipped_reason": "Rows are skipped only if label2_int is missing in the processed LIAR2 binary file.",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0,
        ),
    }

    predictions_frame = usable[
        [
            "id",
            "statement",
            "label_name",
            "label_binary",
            "label2_int",
        ]
    ].copy()

    predictions_frame["prediction_int"] = y_pred
    predictions_frame["prediction_label"] = np.where(
        predictions_frame["prediction_int"] == 1,
        "real",
        "fake",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "binary_bert_metadata_liar2_test_metrics.json"
    summary_path = OUTPUT_DIR / "binary_bert_metadata_liar2_test_summary.txt"
    predictions_path = OUTPUT_DIR / "binary_bert_metadata_liar2_test_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_lines = [
        "LIAR2 evaluation of best project model",
        "",
        f"model: {metrics['model']}",
        f"split: {metrics['split']}",
        f"rows_raw: {metrics['rows_raw']}",
        f"rows_evaluated: {metrics['rows_evaluated']}",
        f"rows_skipped: {metrics['rows_skipped']}",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"f1_macro: {metrics['f1_macro']:.4f}",
        f"f1_weighted: {metrics['f1_weighted']:.4f}",
        "",
        "confusion_matrix:",
        json.dumps(metrics["confusion_matrix"]),
    ]

    with open(summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))

    predictions_frame.to_csv(predictions_path, index=False)

    print("\nLIAR2 evaluation complete.")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print()
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {predictions_path}")

    return metrics


def main():
    evaluate_binary_bert_metadata_on_liar2()


if __name__ == "__main__":
    main()