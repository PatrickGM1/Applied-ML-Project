"""Final evaluation for BERT text-only models (train+valid → test).

Mirrors fake_news_detection/scripts/final_text_metadata_test.py but for Model 1 (BERT text only).

Run (from repo root):
  python -m fake_news_detection.scripts.final_bert_text_only_test
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "final"

TEXT_COLUMN_PREFERRED = "statement"
TEXT_COLUMN_FALLBACK = "statement_clean"


EXPERIMENTS = {
    "multiclass_bert_text_only_final": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file": "test.processed.csv",
        "label_column": "label6_int",
    },
    "binary_bert_text_only_final": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file": "test_binary.processed.csv",
        "label_column": "label2_int",
    },
}


def _get_text_column(frame: pd.DataFrame) -> str:
    if TEXT_COLUMN_PREFERRED in frame.columns:
        return TEXT_COLUMN_PREFERRED
    return TEXT_COLUMN_FALLBACK


def load_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(CLEANED_TEXT_DIR / file_name)


def load_and_concat(files: list[str]) -> pd.DataFrame:
    frames = [load_dataframe(name) for name in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def truncate_frame(frame: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    if not limit or limit <= 0 or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


@dataclass
class TextDataset(torch.utils.data.Dataset):
    texts: list[str]
    labels: list[int]
    tokenizer: any
    max_length: int

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "dataset": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "labels": sorted(pd.Series(y_true).astype(str).unique().tolist()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def save_results(name: str, metrics: dict, train_rows: int, test_rows: int, label_column: str, model_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        **metrics,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "label_column": label_column,
        "model_name": model_name,
    }

    json_path = OUTPUT_DIR / f"{name}_metrics.json"
    txt_path = OUTPUT_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    summary_lines = [
        f"dataset: {payload['dataset']}",
        f"model_name: {payload['model_name']}",
        f"label_column: {payload['label_column']}",
        f"train_rows: {payload['train_rows']}",
        f"test_rows: {payload['test_rows']}",
        f"accuracy: {payload['accuracy']:.4f}",
        f"f1_macro: {payload['f1_macro']:.4f}",
        f"labels: {', '.join(payload['labels'])}",
        "",
        "confusion_matrix:",
        json.dumps(payload["confusion_matrix"]),
    ]
    with open(txt_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final BERT text-only evaluation (train+valid→test).")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="HF model id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    model_name: str,
    batch_size: int,
    epochs: float,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict:
    text_column = _get_text_column(train_df)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    num_labels = int(pd.concat([train_df[label_column], test_df[label_column]]).nunique())
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_texts = train_df[text_column].fillna("").astype(str).tolist()
    test_texts = test_df[text_column].fillna("").astype(str).tolist()
    train_labels = train_df[label_column].astype(int).tolist()
    test_labels = test_df[label_column].astype(int).tolist()

    train_dataset = TextDataset(train_texts, train_labels, tokenizer=tokenizer, max_length=max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer=tokenizer, max_length=max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_tmp" / name),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    pred_output = trainer.predict(test_dataset)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=-1)

    metrics = evaluate_predictions(name, y_true=y_true, y_pred=y_pred)
    save_results(
        name=name,
        metrics=metrics,
        train_rows=len(train_df),
        test_rows=len(test_df),
        label_column=label_column,
        model_name=model_name,
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    for name, config in EXPERIMENTS.items():
        train_df = load_and_concat(config["train_files"])
        test_df = load_dataframe(config["test_file"])

        train_df = truncate_frame(train_df, args.max_train_samples, seed=args.seed)
        test_df = truncate_frame(test_df, args.max_test_samples, seed=args.seed)

        metrics = run_experiment(
            name=name,
            train_df=train_df,
            test_df=test_df,
            label_column=config["label_column"],
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
