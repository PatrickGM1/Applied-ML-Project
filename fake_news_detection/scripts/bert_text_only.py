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

import copy
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
from sklearn.utils.class_weight import compute_class_weight

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
NUM_EPOCHS   = 5
LEARNING_RATE = 1e-5
LEARNING_RATES    = [1e-5, 2e-5, 5e-5]   # LR candidates for the search phase (unused — best LR hardcoded above)
LOG_EVERY_N_STEPS   = 50                   # print training loss every N gradient steps
FREEZE_EPOCHS      = 1                    # epochs to train head-only (BERT encoder frozen)
UNFREEZE_LR_FACTOR = 1.0                  # LR multiplier when unfreezing BERT (1e-5 is already conservative, no reduction needed)
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
        "val_file":    "valid.processed.csv",
        "test_file":   "test.processed.csv",
        "label_column": "label6_int",
        "num_labels":   6,
    },
    "binary_bert_text_only": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "val_file":    "valid_binary.processed.csv",
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
def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, epoch=None, lr=None):
    # Enable training mode: dropout is active and gradients are tracked.
    model.train()
    total_loss = 0.0
    step_losses = []
    n_steps = len(loader)

    for step, batch in enumerate(loader, 1):
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

        step_loss = loss.item()
        total_loss += step_loss
        step_losses.append(step_loss)

        # Print loss every LOG_EVERY_N_STEPS so we can see whether the model
        # is converging or diverging during training (per the TA's feedback).
        if step % LOG_EVERY_N_STEPS == 0 or step == n_steps:
            tag = f"lr={lr:.0e} " if lr is not None else ""
            ep  = f"epoch={epoch} " if epoch is not None else ""
            print(f"    [{tag}{ep}step {step:4d}/{n_steps}]  train_loss={step_loss:.4f}")

    return total_loss / n_steps, step_losses


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


@torch.no_grad()
def eval_loss(model, loader, device, criterion) -> float:
    """Compute average cross-entropy loss on a data split without weight updates.

    Used in the LR search phase to compare how well each learning rate candidate
    generalises to the validation split.
    """
    model.eval()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        total_loss += criterion(logits, labels).item()
    return total_loss / len(loader)


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


def run_lr_search(name: str, config: dict) -> float:
    """Try all candidates in LEARNING_RATES and return the one with lowest final val loss.

    Trains on the single train split only (train_files[0]) and validates on
    val_file so the test set is never touched during LR selection.  The final
    experiment then uses the best LR with train + valid combined.
    """
    print(f"\n{'='*60}")
    print(f"LR search: {name}  candidates={LEARNING_RATES}")
    print(f"{'='*60}")

    train_df = pd.read_csv(CLEANED_TEXT_DIR / config["train_files"][0])
    val_df   = pd.read_csv(CLEANED_TEXT_DIR / config["val_file"])

    label_column = config["label_column"]
    num_labels   = config["num_labels"]
    text_column  = _pick_text_column(train_df)

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds     = TextDataset(train_df, tokenizer, text_column, label_column, MAX_LENGTH)
    val_ds       = TextDataset(val_df,   tokenizer, text_column, label_column, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    lr_results: dict[float, float] = {}  # lr -> final-epoch val loss

    for lr in LEARNING_RATES:
        print(f"\n  --- lr={lr:.0e} ---")
        set_seed(RANDOM_SEED)  # same initialisation for every LR so results are comparable
        model     = BertTextOnly(MODEL_NAME, num_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        total_steps  = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            avg_train, _ = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, criterion,
                epoch=epoch, lr=lr,
            )
            val = eval_loss(model, val_loader, device, criterion)
            print(f"    epoch {epoch}/{NUM_EPOCHS}  avg_train_loss={avg_train:.4f}  val_loss={val:.4f}")

        lr_results[lr] = val  # compare by final-epoch val loss

    print(f"\n  LR comparison for {name}:")
    print(f"  {'LR':>8}  {'val_loss':>10}")
    for lr, vl in lr_results.items():
        marker = "  <-- best" if lr == min(lr_results, key=lr_results.__getitem__) else ""
        print(f"  {lr:8.0e}  {vl:10.4f}{marker}")

    best_lr = min(lr_results, key=lr_results.__getitem__)
    print(f"\n  → best lr = {best_lr:.0e}  (val_loss={lr_results[best_lr]:.4f})")
    return best_lr


# Runs one complete experiment: load data, train model, evaluate, save outputs.
def run_experiment(name: str, config: dict, lr: float = LEARNING_RATE) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # --- Load data ---
    # Train on train split only so val_file is a clean, unseen monitor signal.
    # Early stopping uses val_loss to decide when to stop — this only works if
    # val data was never seen during training.
    train_df = pd.read_csv(CLEANED_TEXT_DIR / config["train_files"][0])
    val_df   = pd.read_csv(CLEANED_TEXT_DIR / config["val_file"])
    test_df  = pd.read_csv(CLEANED_TEXT_DIR / config["test_file"])

    label_column = config["label_column"]
    num_labels   = config["num_labels"]
    text_column  = _pick_text_column(train_df)

    print(f"  Train rows : {len(train_df)}")
    print(f"  Val rows   : {len(val_df)}")
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
    val_dataset = TextDataset(
        df=val_df,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
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

    # Class weights only for multiclass — binary is balanced so weighting hurts accuracy.
    train_labels = train_df[label_column].astype(int).values
    if num_labels > 2:
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_labels), y=train_labels
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Phase 1: train head only (BERT encoder frozen) ---
    # Freezing BERT lets the new classifier head reach a stable starting point
    # before any gradients flow back into the pretrained weights (point 3).
    print(f"\n  [Phase 1] Freezing BERT encoder — training head only for {FREEZE_EPOCHS} epoch(s)")
    for param in model.bert.parameters():
        param.requires_grad = False

    head_params     = [p for p in model.parameters() if p.requires_grad]
    optimizer_p1    = torch.optim.AdamW(head_params, lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps_p1  = len(train_loader) * FREEZE_EPOCHS
    warmup_steps_p1 = int(total_steps_p1 * WARMUP_RATIO)
    scheduler_p1    = get_linear_schedule_with_warmup(
        optimizer_p1,
        num_warmup_steps=warmup_steps_p1,
        num_training_steps=total_steps_p1,
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, FREEZE_EPOCHS + 1):
        avg_train, _ = train_one_epoch(
            model, train_loader, optimizer_p1, scheduler_p1, device, criterion,
            epoch=epoch, lr=lr,
        )
        v_loss = eval_loss(model, val_loader, device, criterion)
        print(f"  [frozen]   Epoch {epoch}/{FREEZE_EPOCHS}  avg_train_loss={avg_train:.4f}  val_loss={v_loss:.4f}")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = copy.deepcopy(model.state_dict())

    # --- Phase 2: unfreeze BERT and fine-tune all layers ---
    # Early stopping (patience=2) saves the best weights and restores them after
    # training ends or stops early — prevents overfitting on longer runs.
    unfreeze_lr      = lr * UNFREEZE_LR_FACTOR
    remaining_epochs = NUM_EPOCHS - FREEZE_EPOCHS
    print(f"\n  [Phase 2] Unfreezing BERT — fine-tuning all layers for up to {remaining_epochs} epoch(s)  lr={unfreeze_lr:.0e}")
    for param in model.bert.parameters():
        param.requires_grad = True

    optimizer_p2    = torch.optim.AdamW(model.parameters(), lr=unfreeze_lr, weight_decay=WEIGHT_DECAY)
    total_steps_p2  = len(train_loader) * remaining_epochs
    warmup_steps_p2 = int(total_steps_p2 * WARMUP_RATIO)
    scheduler_p2    = get_linear_schedule_with_warmup(
        optimizer_p2,
        num_warmup_steps=warmup_steps_p2,
        num_training_steps=total_steps_p2,
    )

    patience   = 2
    no_improve = 0

    for epoch in range(FREEZE_EPOCHS + 1, NUM_EPOCHS + 1):
        avg_train, _ = train_one_epoch(
            model, train_loader, optimizer_p2, scheduler_p2, device, criterion,
            epoch=epoch, lr=unfreeze_lr,
        )
        v_loss = eval_loss(model, val_loader, device, criterion)
        print(f"  [unfrozen] Epoch {epoch}/{NUM_EPOCHS}  avg_train_loss={avg_train:.4f}  val_loss={v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs)")
                break

    print(f"  Restoring best weights (val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state)

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
        name: run_experiment(name, config, lr=LEARNING_RATE)
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