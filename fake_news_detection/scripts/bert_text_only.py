"""Model 1: BERT text-only for fake news detection.
----------------------------
This script trains a BERT classifier that uses ONLY the text statement.
It is the baseline model in the project. 

The high-level pipeline is:

    raw statement text
        -> BERT tokenizer
        -> BERT encoder
        -> [CLS] embedding, a 768-number summary vector
        -> small neural-network classifier
        -> predicted fake-news label


Architecture
------------
- BERT encoder  →  [CLS] embedding  (768-dim)
- Classifier: [CLS] vector  →  Linear → ReLU → Dropout → Linear → logits

Run:
    python fake_news_detection/scripts/bert_text_only.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# DataLoader groups many Dataset items into mini-batches for training.
from torch.utils.data import Dataset, DataLoader
# Hugging Face Transformers provides the pretrained BERT tokenizer/model
# and the linear learning-rate scheduler with warm-up.
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed,
)



# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
# This section only defines where the script expects to find data and where it
# will save outputs. It does not train anything yet.
#
# Expected project layout:
#   fake_news_detection/
#       scripts/bert_text_v2.py
#       data/processed/cleaned_text/*.csv
#       artifacts/final/bert_text_only/      <- metrics go here
#       artifacts/models/bert_text_only/     <- trained weights go here

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"

METRICS_DIR = PROJECT_DIR / "artifacts" / "final" / "bert_text_only"  #  <- metrics go here
MODELS_DIR  = PROJECT_DIR / "artifacts" / "models" / "bert_text_only" #  <- trained weights go here


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------


# Pretrained model downloaded from Hugging Face.
MODEL_NAME   = "bert-base-uncased"

MAX_LENGTH   = 128          # Max nr of BERT tokens per statement. Longer texts are truncated;
                            # shorter texts are padded to exactly this length.
RANDOM_SEED  = 42
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01         # Regularization term that discourages overly large weights
WARMUP_RATIO  = 0.1          # fraction of total steps used for LR warm-up
DROPOUT_RATE  = 0.3          # helps reduce overfitting in the classifier/fusion head
HIDDEN_DIM    = 256          # size of the hidden layer in the fusion head


# BERT works best with raw text because its own tokenizer handles splitting words
# into wordpieces. 
TEXT_COLUMN_PREFERRED = "statement"        # raw text — BERT tokenises itself
TEXT_COLUMN_FALLBACK  = "statement_clean"

# The script runs two experiments automatically:
#   1. multiclass: predicts one of 6 truthfulness labels
#   2. binary: predicts fake vs real / false vs true depending on your preprocessing
EXPERIMENTS = {
    "multiclass_bert_text_only": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file":   "test.processed.csv",
        "label_column": "label6_int",
        "num_labels":   6,
    },
    "binary_bert_text_only": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file":   "test_binary.processed.csv",
        "label_column": "label2_int",
        "num_labels":   2,
    },
}

set_seed(RANDOM_SEED)
# Avoids noisy warnings 
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# 1.  Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Converts CSV rows into tensors that BERT can consume.

    A PyTorch Dataset behaves like a list -> when DataLoader asks for item `idx`,
    `__getitem__` returns one training example. In this script, one example is:
        statement text -> tokenized BERT inputs
        label          -> integer class label

    The model never sees raw strings directly. It sees token IDs, attention masks,
    token type IDs, and labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_column: str,
        label_column: str,
        max_length: int,
    ):
    
        self.texts    = df[text_column].fillna("").astype(str).tolist()
        # Store labels as integers because CrossEntropyLoss expects integer class IDs.
        self.labels   = df[label_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts) # how many examples are in the dataset

    def __getitem__(self, idx):
        # Tokenize one text example.
        # truncation=True: cut text if it is longer than MAX_LENGTH.
        # padding="max_length": pad all examples to the same length so they can form a batch.
        # return_tensors="pt": return PyTorch tensors.
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # BERT receives input_ids and attention_mask.
        # token_type_ids are included for consistency with the metadata script.
        # labels are used only during training/evaluation to compute the loss/metrics.
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 2.  Text-only model
# ---------------------------------------------------------------------------

class BertTextOnly(nn.Module):
    """
    The neural network used for the text-only BERT model.

    This is the most important class in the file. It explicitly defines the
    architecture 

    BERT encoder  →  classifier head  →  classifier.

    Text path  : BERT [CLS] token  →  768-dim vector
    Classifier : Linear(768, HIDDEN_DIM)  →  ReLU
                 →  Dropout  →  Linear(HIDDEN_DIM, num_labels)
    """

    def __init__(self, bert_model_name: str, num_labels: int):
        super().__init__()
        # Load the pretrained BERT encoder without a built-in classification head.
        # We add our own classifier below so the text-only model mirrors the
        # BERT + metadata model as closely as possible.
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # For bert-base-uncased, each token representation has 768 dimensions.
        bert_dim  = self.bert.config.hidden_size          # 768 for bert-base

        # This classifier receives the 768-dimensional [CLS] vector and outputs
        # one score per class. These raw scores are called logits.
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Forward pass through BERT.
        # Output shape of last_hidden_state is: (batch_size, sequence_length, 768).
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] representation — shape (batch, 768).
        # The first token position, index 0, corresponds to BERT's special [CLS] token.
        # For classification tasks, this vector is commonly used as a summary of the text.
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Return logits, not probabilities. CrossEntropyLoss expects raw logits.
        return self.classifier(cls_embedding)          # (batch, num_labels)


# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------

# One epoch means one complete pass through the training set.
def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    # Enable training mode: dropout is active and gradients are tracked.
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Each batch is a dictionary created by TextDataset and grouped by DataLoader.
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        # Remove gradients from the previous batch. PyTorch accumulates gradients by default.
        optimizer.zero_grad()
        # Forward pass: get model predictions as logits.
        logits = model(input_ids, attention_mask, token_type_ids)

        # Compare predictions with true labels.
        # CrossEntropyLoss combines softmax + negative log likelihood internally.
        loss   = criterion(logits, labels)
        # Backpropagation: compute gradients for all trainable parameters.
        loss.backward()

        # Gradient clipping — standard practice for fine-tuning BERT
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights using gradients.
        optimizer.step()

        # Update learning rate according to the warm-up/decay schedule.
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# During evaluation we do not want to compute gradients, so torch.no_grad() saves memory.
@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    # Evaluation mode: dropout is disabled, so predictions are deterministic for a fixed model.
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"]

        # Forward pass on the test batch.
        logits = model(input_ids, attention_mask, token_type_ids)

        # Select the class with the highest logit score for each example.
        preds  = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# 4.  Main experiment runner
# ---------------------------------------------------------------------------

# Prefer raw statements when available, otherwise fall back to cleaned text.
def _pick_text_column(df: pd.DataFrame) -> str:
    return (
        TEXT_COLUMN_PREFERRED
        if TEXT_COLUMN_PREFERRED in df.columns
        else TEXT_COLUMN_FALLBACK
    )


# Runs one complete experiment: load data, train model, evaluate, save outputs.
def run_experiment(name: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # --- Load data ---
    # Training uses train + validation combined. The test set is held out until final evaluation.
    train_dfs = [
        pd.read_csv(CLEANED_TEXT_DIR / f)
        for f in config["train_files"]
    ]
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.read_csv(CLEANED_TEXT_DIR / config["test_file"])

    label_column = config["label_column"]
    num_labels   = config["num_labels"]
    text_column  = _pick_text_column(train_df)

    print(f"  Train rows : {len(train_df)}")
    print(f"  Test rows  : {len(test_df)}")
    print(f"  Text column: {text_column}")
    print(f"  Labels     : {num_labels}")

    # --- Tokenizer ---
    # The tokenizer converts raw text into BERT token IDs.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextDataset(
        df=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=MAX_LENGTH,
    )
    test_dataset = TextDataset(
        df=test_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )

    # --- Device ---
    # Use GPU when available. Otherwise, run on CPU, which is much slower for BERT.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Model ---
    # Create a fresh model for this experiment. The multiclass and binary experiments
    # are trained separately because they have different numbers of output labels.
    model = BertTextOnly(
        bert_model_name=MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    # --- Optimiser & scheduler ---
    # AdamW is the standard optimizer for fine-tuning transformer models.
    # The scheduler first warms up the learning rate, then linearly decays it.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

# CrossEntropyLoss is used for multi-class classification, including binary
    # classification when the model outputs 2 logits.
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={avg_loss:.4f}")

    # --- Final evaluation on held-out test set ---
    # Important: the test set is not used during training. It is used once here.
    predictions, y_true = predict(model, test_loader, device)

    # Store all important evaluation results in a dictionary so they can be saved
    # as JSON and summarized in a text file.
    metrics = {
        "dataset":      name,
        "model_name":   MODEL_NAME,
        "text_column":  text_column,
        "label_column": label_column,
        "num_labels":   int(num_labels),
        "max_length":   int(MAX_LENGTH),
        "train_rows":   int(len(train_df)),
        "eval_rows":    int(len(test_df)),
        "accuracy":     float(accuracy_score(y_true, predictions)),
        "f1_macro":     float(f1_score(y_true, predictions, average="macro",    zero_division=0)),
        "f1_weighted":  float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "labels":       sorted(pd.Series(y_true).astype(str).unique().tolist()),
        "confusion_matrix":      confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }

    print(
        f"\n  → accuracy={metrics['accuracy']:.4f}  "
        f"f1_macro={metrics['f1_macro']:.4f}  "
        f"f1_weighted={metrics['f1_weighted']:.4f}"
    )

    # --- Save metrics ---
    _save_results(name, metrics)

    # --- Save model weights ---
    model_dir = MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model_weights.pt")
    tokenizer.save_pretrained(str(model_dir))

    return metrics


# Save both a detailed JSON file and a smaller human-readable TXT summary.
def _save_results(name: str, metrics: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = METRICS_DIR / f"{name}_metrics.json"
    txt_path  = METRICS_DIR / f"{name}_summary.txt"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

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

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Entry point. This function runs both experiments defined in EXPERIMENTS.
def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {MODEL_NAME}")
    print(f"Data dir   : {CLEANED_TEXT_DIR}")
    print(f"Metrics dir: {METRICS_DIR}")
    print(f"Models dir : {MODELS_DIR}")

    if torch.cuda.is_available():
        print(f"CUDA: yes  —  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: no  (training will be slow on CPU)")

    results = {
        name: run_experiment(name, config)
        for name, config in EXPERIMENTS.items()
    }

    print("\n" + "="*60)
    print("Final results")
    print("="*60)
    for name, m in results.items():
        print(
            f"  {name}\n"
            f"    accuracy={m['accuracy']:.4f}  "
            f"f1_macro={m['f1_macro']:.4f}  "
            f"f1_weighted={m['f1_weighted']:.4f}"
        )


if __name__ == "__main__":
    main()
