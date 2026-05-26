"""Final-style BERT + metadata fusion (train+valid → test).

Model idea (CS view):
- Text encoder: a pretrained Transformer (BERT/DistilBERT/etc.) turns a statement into a dense vector.
- Metadata encoder: tabular/categorical metadata is turned into a numeric feature vector.
- Fusion: concatenate [text_vector | metadata_vector] and learn a classifier on top.

This script reuses the project’s existing metadata preprocessing from
fake_news_detection/features/metadata.py.

Run (from repo root):
  python -m fake_news_detection.scripts.bert_text_metadata

Tips:
  - Use --max-train-samples / --max-test-samples for a quick smoke test.
  - Use --save-model if you want to persist the trained fusion model + encoders.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from fake_news_detection.features.metadata import fit_metadata_transformers, transform_metadata


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"
METRICS_DIR = PROJECT_DIR / "artifacts" / "final"
MODELS_DIR = PROJECT_DIR / "artifacts" / "models" / "bert_text_metadata_fusion"

TEXT_COLUMN_PREFERRED = "statement"  # raw statement if present
TEXT_COLUMN_FALLBACK = "statement_clean"


EXPERIMENTS = {
    "multiclass_bert_text_metadata_fusion": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file": "test.processed.csv",
        "label_column": "label6_int",
        "num_labels": 6,
    },
    "binary_bert_text_metadata_fusion": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file": "test_binary.processed.csv",
        "label_column": "label2_int",
        "num_labels": 2,
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


def compute_metrics_simple(eval_pred) -> dict:
    """Trainer-time metrics (kept small)."""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "dataset": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "labels": sorted(pd.Series(y_true).astype(str).unique().tolist()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def save_results(name: str, payload: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = METRICS_DIR / f"{name}_metrics.json"
    txt_path = METRICS_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    summary_lines = [
        f"dataset: {payload['dataset']}",
        f"model_name: {payload['model_name']}",
        f"label_column: {payload['label_column']}",
        f"train_rows: {payload['train_rows']}",
        f"test_rows: {payload['test_rows']}",
        f"meta_dim: {payload['meta_dim']}",
        f"accuracy: {payload['accuracy']:.4f}",
        f"f1_macro: {payload['f1_macro']:.4f}",
        f"labels: {', '.join(payload['labels'])}",
        "",
        "confusion_matrix:",
        json.dumps(payload["confusion_matrix"]),
    ]
    with open(txt_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))


@dataclass
class TextMetadataDataset(torch.utils.data.Dataset):
    """Each item = tokenized text + dense metadata vector + label."""

    texts: list[str]
    metadata: np.ndarray  # shape: [N, meta_dim]
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
        encoded["metadata"] = torch.from_numpy(self.metadata[idx]).float()
        encoded["labels"] = int(self.labels[idx])
        return encoded


class BertMetadataFusionModel(nn.Module):
    """Late-fusion classifier: [CLS text embedding] + [metadata projection] -> logits."""

    def __init__(
        self,
        model_name: str,
        meta_dim: int,
        num_labels: int,
        meta_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        text_hidden = int(self.text_model.config.hidden_size)

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_hidden + meta_hidden, num_labels),
        )
        self.num_labels = int(num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: dict,
    ) -> SequenceClassifierOutput:
        # Text encoding (BERT): last_hidden_state[:, 0, :] is the CLS token embedding.
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if metadata is None:
            raise ValueError("Missing required 'metadata' tensor in the batch")

        meta_embedding = self.meta_proj(metadata)
        fused = torch.cat([cls_embedding, meta_embedding], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def _build_training_args(output_dir: Path, args: argparse.Namespace) -> TrainingArguments:
    # Transformers renamed `evaluation_strategy` -> `eval_strategy` in newer versions.
    kwargs: dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "report_to": [],
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
    }

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "no"
    else:
        kwargs["eval_strategy"] = "no"

    return TrainingArguments(**kwargs)


def _build_trainer(
    model: nn.Module,
    training_args: TrainingArguments,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    tokenizer,
) -> Trainer:
    # Collator pads text fields to the longest sequence in the batch.
    text_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(batch: list[dict]) -> dict:
        # Split into text-like fields and our extra "metadata" tensor.
        metadata = torch.stack([item.pop("metadata") for item in batch], dim=0)
        batch_out = text_collator(batch)
        batch_out["metadata"] = metadata
        return batch_out

    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": collate,
        "compute_metrics": compute_metrics_simple,
    }

    # Transformers versions differ on whether Trainer accepts `tokenizer` or `processing_class`.
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


def _save_model_bundle(
    name: str,
    model: BertMetadataFusionModel,
    tokenizer,
    meta_transformers,
    meta_dim: int,
    args: argparse.Namespace,
) -> None:
    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Torch weights
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": args.model_name,
            "meta_dim": int(meta_dim),
            "meta_hidden": int(args.meta_hidden),
            "num_labels": int(args.num_labels_override) if args.num_labels_override else None,
            "max_length": int(args.max_length),
        },
        out_dir / "model.pt",
    )

    # 2) Tokenizer
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    # 3) Metadata encoders (sklearn objects)
    joblib.dump(meta_transformers, out_dir / "metadata_transformers.joblib")


def run_experiment(name: str, config: dict, args: argparse.Namespace) -> dict:
    train_df = load_and_concat(config["train_files"])
    test_df = load_dataframe(config["test_file"])

    train_df = truncate_frame(train_df, args.max_train_samples, seed=args.seed)
    test_df = truncate_frame(test_df, args.max_test_samples, seed=args.seed)

    text_column = _get_text_column(train_df)

    # Metadata preprocessing: reuse the same exact encoders as the TF-IDF+metadata baselines.
    x_train_meta_sparse, meta_transformers = fit_metadata_transformers(train_df)
    x_test_meta_sparse = transform_metadata(test_df, meta_transformers)

    # For simplicity we use dense metadata in the fusion model.
    x_train_meta = x_train_meta_sparse.toarray().astype(np.float32)
    x_test_meta = x_test_meta_sparse.toarray().astype(np.float32)
    meta_dim = int(x_train_meta.shape[1])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_texts = train_df[text_column].fillna("").astype(str).tolist()
    test_texts = test_df[text_column].fillna("").astype(str).tolist()
    train_labels = train_df[config["label_column"]].astype(int).tolist()
    test_labels = test_df[config["label_column"]].astype(int).tolist()

    train_dataset = TextMetadataDataset(
        texts=train_texts,
        metadata=x_train_meta,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = TextMetadataDataset(
        texts=test_texts,
        metadata=x_test_meta,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Reduce noisy parallel tokenizers warnings on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    num_labels = int(config["num_labels"])
    if args.num_labels_override:
        num_labels = int(args.num_labels_override)

    model = BertMetadataFusionModel(
        model_name=args.model_name,
        meta_dim=meta_dim,
        num_labels=num_labels,
        meta_hidden=args.meta_hidden,
        dropout=args.dropout,
    )

    training_args = _build_training_args(METRICS_DIR / "_tmp" / name, args)
    trainer = _build_trainer(model, training_args, train_dataset, test_dataset, tokenizer)

    trainer.train()

    pred_output = trainer.predict(test_dataset)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=-1)

    metrics = evaluate_predictions(name, y_true=y_true, y_pred=y_pred)

    payload = {
        **metrics,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_column": config["label_column"],
        "model_name": args.model_name,
        "text_column": text_column,
        "meta_dim": int(meta_dim),
        "fusion": "concat(cls_embedding, meta_projection)",
        "meta_hidden": int(args.meta_hidden),
        "max_length": int(args.max_length),
        "epochs": float(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
    }
    save_results(name, payload)

    if args.save_model:
        _save_model_bundle(name, model, tokenizer, meta_transformers, meta_dim, args)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model 2: BERT + metadata fusion (train+valid→test).")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="HF model id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    # Fusion-specific knobs
    parser.add_argument("--meta-hidden", type=int, default=128, help="Size of metadata projection")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save-model", action="store_true")

    # Escape hatch if your label columns don't match the expected 2/6 setup.
    parser.add_argument("--num-labels-override", type=int, default=0)

    args = parser.parse_args()
    if args.num_labels_override <= 0:
        args.num_labels_override = 0
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results = {}
    for name, config in EXPERIMENTS.items():
        results[name] = run_experiment(name, config, args)
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, f1_macro={results[name]['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
