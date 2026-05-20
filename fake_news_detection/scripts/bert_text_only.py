"""Final-style BERT text-only training/evaluation (train+valid → test).

Implements "Model 1: BERT (text only)".

High-level flow:
- Load processed CSVs (train + valid + test)
- Fine-tune a pretrained BERT model on train+valid
- Evaluate once on the held-out test split
- Save metrics (JSON + TXT) + optionally the trained model

Run (from repo root):
    python3 fake_news_detection/scripts/bert_v3.py
    python3 fake_news_detection/scripts/train_bert_text_only.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"

# Match the rest of the repo: final evaluation metrics go under artifacts/final/
METRICS_DIR = PROJECT_DIR / "artifacts" / "final"

# Keep trained weights separate (useful if you want to reuse without retraining).
MODELS_DIR = PROJECT_DIR / "artifacts" / "models" / "bert_text_only"

MODEL_NAME = "bert-base-uncased"

TEXT_COLUMN_PREFERRED = "statement"  # raw statement if present
TEXT_COLUMN_FALLBACK = "statement_clean"

MAX_LENGTH = 128 
RANDOM_SEED = 42

EXPERIMENTS = {
    "multiclass_bert_text_only": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file": "test.processed.csv",
        "label_column": "label6_int",
        "num_labels": 6,
    },
    "binary_bert_text_only": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file": "test_binary.processed.csv",
        "label_column": "label2_int",
        "num_labels": 2,
    },
}


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

set_seed(RANDOM_SEED)


def _pick_text_column(df: pd.DataFrame) -> str:
    """Prefer the raw statement column when available."""

    if TEXT_COLUMN_PREFERRED in df.columns:
        return TEXT_COLUMN_PREFERRED
    return TEXT_COLUMN_FALLBACK


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_dataframe(file_name):
    file_path = CLEANED_TEXT_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find file: {file_path}\n"
            f"Check CLEANED_TEXT_DIR and the file names in EXPERIMENTS."
        )

    return pd.read_csv(file_path)


def validate_columns(df, text_column, label_column, file_description):
    missing_columns = []

    if text_column not in df.columns:
        missing_columns.append(text_column)

    if label_column not in df.columns:
        missing_columns.append(label_column)

    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in {file_description}: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )


class BertTextDataset(Dataset):
    """Torch Dataset that turns one row into BERT inputs.

    Key idea: the tokenizer converts text to token IDs (+ attention mask). We pad
    to a fixed length here to keep tensor shapes consistent.
    """

    def __init__(self, dataframe, tokenizer, text_column, label_column, max_length):
        self.texts = dataframe[text_column].fillna("").astype(str).tolist()
        self.labels = dataframe[label_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }

        return item


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
    }


# --------------------------------------------------
# Training arguments
# --------------------------------------------------

def build_training_args(name):
    # Trainer needs an output directory even when we don't save checkpoints.
    output_dir = MODELS_DIR / name / "checkpoints"

    return TrainingArguments(
        output_dir=str(output_dir),
        
        # Important: we don't evaluate on the test set during training.
        # We only evaluate once at the end on the held-out test split.
        eval_strategy="no",
        save_strategy="no",

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,

        # Since save_strategy="no", we cannot load the best checkpoint at the end.
        load_best_model_at_end=False,


        logging_steps=50,

        report_to="none",

        seed=RANDOM_SEED,
    )


# --------------------------------------------------
# Training/evaluation
# --------------------------------------------------

def evaluate_split(name, train_df, eval_df, label_column, num_labels):
    # Choose which text column to use based on what's in the processed CSV.
    text_column = _pick_text_column(train_df)

    validate_columns(
        train_df,
        text_column=text_column,
        label_column=label_column,
        file_description=f"{name} training data",
    )

    validate_columns(
        eval_df,
        text_column=text_column,
        label_column=label_column,
        file_description=f"{name} evaluation data",
    )

    # Reduce noisy parallel tokenizers warnings on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BertTextDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=MAX_LENGTH,
    )

    eval_dataset = BertTextDataset(
        dataframe=eval_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    training_args = build_training_args(name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # No eval_dataset: final evaluation happens once via trainer.predict below.

        # Newer transformers versions prefer processing_class.
        # If your version complains, replace this with: tokenizer=tokenizer
        processing_class=tokenizer,
    )

    trainer.train()

    # Final test evaluation (held-out split).
    predictions_output = trainer.predict(eval_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    y_eval = eval_df[label_column].astype(int).to_numpy()

    metrics = {
        "dataset": name,
        "model_name": MODEL_NAME,
        "text_column": text_column,
        "label_column": label_column,
        "num_labels": int(num_labels),
        "max_length": int(MAX_LENGTH),
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "f1_macro": float(
            f1_score(
                y_eval,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_eval,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "labels": sorted(pd.Series(y_eval).astype(str).unique().tolist()),
        "confusion_matrix": confusion_matrix(y_eval, predictions).tolist(),
        "classification_report": classification_report(
            y_eval,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save trained weights so you can reuse the model without retraining.
    model_dir = MODELS_DIR / name / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(
        str(model_dir),
        safe_serialization=False,
    )

    tokenizer.save_pretrained(str(model_dir))

    return metrics


# --------------------------------------------------
# Saving results
# --------------------------------------------------

def save_results(name, metrics):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = METRICS_DIR / f"{name}_metrics.json"
    txt_path = METRICS_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_lines = [
        f"dataset: {metrics['dataset']}",
        f"model_name: {metrics['model_name']}",
        f"text_column: {metrics['text_column']}",
        f"label_column: {metrics['label_column']}",
        f"num_labels: {metrics['num_labels']}",
        f"max_length: {metrics['max_length']}",
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

    with open(txt_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))


# --------------------------------------------------
# Experiment loop
# --------------------------------------------------

def run_experiment(name, config):
    train_dfs = [
        load_dataframe(file_name)
        for file_name in config["train_files"]
    ]

    train_df = pd.concat(train_dfs, ignore_index=True)
    eval_df = load_dataframe(config["test_file"])

    metrics = evaluate_split(
        name=name,
        train_df=train_df,
        eval_df=eval_df,
        label_column=config["label_column"],
        num_labels=config["num_labels"],
    )

    save_results(name, metrics)

    return metrics


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {MODEL_NAME}")
    print(f"Reading data from: {CLEANED_TEXT_DIR}")
    print(f"Writing metrics to: {METRICS_DIR}")
    print(f"Writing models to: {MODELS_DIR}")

    if torch.cuda.is_available():
        print(f"CUDA available: yes")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: no")
        print("Warning: BERT training will be slow on CPU.")

    results = {
        name: run_experiment(name, config)
        for name, config in EXPERIMENTS.items()
    }

    print("\nFinal results:")

    for name, metrics in results.items():
        print(
            f"{name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )


if __name__ == "__main__":
    main()