"""Model 2: BERT + Metadata Fusion for fake news detection.
----------------------------
This script trains a BERT classifier that uses BOTH:

    1. the text statement
    2. extra metadata about the statement/speaker/context

The high-level pipeline is:

    raw statement text
        -> BERT tokenizer
        -> BERT encoder
        -> [CLS] embedding, a 768-number text summary vector

    metadata columns
        -> sklearn preprocessing
        -> one numerical metadata vector

    [CLS] text vector + metadata vector
        -> fusion classifier
        -> predicted fake-news label

This is called metadata fusion because the model joins two information sources:
the BERT text representation + the preprocessed metadata vector.


Architecture
------------
- BERT encoder  →  [CLS] embedding  (768-dim)
- Metadata branch (same pipeline as TF-IDF scripts):
    categorical columns  →  LabelEncoder  →  one-hot  (via sklearn)
    numeric columns (hist1-5)  →  StandardScaler
    multi-value columns (subjects) →  MultiLabelBinarizer
  All metadata features are concatenated into a flat numpy vector.
- Fusion: [CLS] vector  ‖  metadata vector  →  Linear → (optional ReLU+Dropout) → Linear → logits
- The metadata branch is a frozen sklearn transform; only the BERT encoder
  and the fusion head are trained end-to-end.

This mirrors exactly what the TF-IDF scripts do for metadata so the two
approaches are directly comparable.

Run:
    python fake_news_detection/scripts/bert_text_metadata.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# - LabelEncoder helps encode categorical values
# - StandardScaler normalizes numeric history counts
# - MultiLabelBinarizer handles columns containing multiple subjects
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer

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
# This section defines where the script expects the processed CSV data and where
# it will save metrics/model weights. No training happens in this section.
#
# Expected project layout:
#   fake_news_detection/
#       scripts/bert_text_metadata.py
#       data/processed/cleaned_text/*.csv
#       artifacts/final/bert_text_metadata/     
#       artifacts/models/bert_text_metadata/     

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "cleaned_text"

METRICS_DIR = PROJECT_DIR / "artifacts" / "final" / "bert_text_metadata"  #  <- metrics go here
MODELS_DIR  = PROJECT_DIR / "artifacts" / "models" / "bert_text_metadata" #  <- trained weights go here


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

# Metadata column definitions.
CATEGORICAL_COLS = ["speaker", "party", "state", "speaker_job", "context"]
NUMERIC_COLS     = ["hist1", "hist2", "hist3", "hist4", "hist5"]
SUBJECTS_COL     = "subjects"

# Use the raw statement column when possible because BERT has its own tokenizer.
TEXT_COLUMN_PREFERRED = "statement"        # raw text - BERT tokenises itself
TEXT_COLUMN_FALLBACK  = "statement_clean"


EXPERIMENTS = {
    "multiclass_bert_metadata": {
        "train_files": ["train.processed.csv", "valid.processed.csv"],
        "test_file":   "test.processed.csv",
        "label_column": "label6_int",
        "num_labels":   6,
    },
    "binary_bert_metadata": {
        "train_files": ["train_binary.processed.csv", "valid_binary.processed.csv"],
        "test_file":   "test_binary.processed.csv",
        "label_column": "label2_int",
        "num_labels":   2,
    },
}

set_seed(RANDOM_SEED)
# Avoids noisy tokenizer warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# 1.  Metadata transformers  (mirrors fit_metadata_transformers / transform_metadata)
# ---------------------------------------------------------------------------

class MetadataTransformers:
    """Fit-once, transform-many sklearn metadata pipeline.

    This class is responsible for turning all metadata columns into one numerical
    matrix. It works in two phases:

    1. fit(train_df)
       Learn the possible categories, scaling parameters, and subject vocabulary
       from the training data only.

    2. transform(test_df)
       Apply the already-learned transformations to new data.
    """

    def __init__(self):
        # One LabelEncoder per categorical column. Each learns the categories
        # that appeared in the training data.
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()  # learns mean/std for numeric metadata columns.
        self.mlb = MultiLabelBinarizer()# learns all possible subject labels.

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        # Fit all metadata preprocessing on the training dataframe and return
        # the final training metadata matrix.
        parts = []

        # --- Categorical columns → integer → one-hot (via get_dummies) ---
        for col in CATEGORICAL_COLS:
            # Learn all categories in this column from training data.
            le = LabelEncoder()
            
            filled = df[col].fillna("unknown").astype(str)
            le.fit(filled)
            self.label_encoders[col] = le
            encoded = le.transform(filled)
            
            ohe = np.zeros((len(df), len(le.classes_)), dtype=np.float32)
            ohe[np.arange(len(df)), encoded] = 1.0
            parts.append(ohe)

        # --- Numeric columns → StandardScaler ---
        num_data = df[NUMERIC_COLS].fillna(0).values.astype(np.float32)
        self.scaler.fit(num_data)
        parts.append(self.scaler.transform(num_data))

        # --- Multi-value subjects → MultiLabelBinarizer ---
        # Parse comma-separated subjects into lists like ["economy", "jobs"].
        subjects_list = _parse_subjects(df)
        self.mlb.fit(subjects_list)
        parts.append(self.mlb.transform(subjects_list).astype(np.float32))

        # Horizontally concatenate all metadata blocks into one matrix:
        # categorical one-hot features + scaled numeric features + subject features.
        # Return final metadata matrix for this dataframe.
        return np.hstack(parts)

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Transform validation/test data using encoders fitted on training data.
        # Prevents data leakage.
        parts = []

        for col in CATEGORICAL_COLS:
            le = self.label_encoders[col]
            filled = df[col].fillna("unknown").astype(str)
            # If a category appears in test but not in train, map it to index 0.
            encoded = np.array(
                [
                    le.transform([v])[0] if v in le.classes_ else 0
                    for v in filled
                ],
                dtype=np.int64,
            )
            ohe = np.zeros((len(df), len(le.classes_)), dtype=np.float32)
            ohe[np.arange(len(df)), encoded] = 1.0
            parts.append(ohe)


        num_data = df[NUMERIC_COLS].fillna(0).values.astype(np.float32)
        parts.append(self.scaler.transform(num_data))

        # Parse comma-separated subjects into lists like ["economy", "jobs"].
        subjects_list = _parse_subjects(df)
        parts.append(self.mlb.transform(subjects_list).astype(np.float32))

        return np.hstack(parts)


# Converts the subjects column from a comma-separated string into a list of tokens.
# Example: "economy,jobs" -> ["economy", "jobs"].
def _parse_subjects(df: pd.DataFrame) -> list[list[str]]:
    result = []
    for val in df[SUBJECTS_COL].fillna(""):
        # Lowercase and strip spaces so " Economy " and "economy" are treated the same.
        tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
        result.append(tokens)
    return result


# ---------------------------------------------------------------------------
# 2.  Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """Converts CSV rows into BERT inputs + metadata tensors.

    This is like TextDataset in the text-only script, but each example also
    returns a metadata vector. One item contains:

        input_ids      -> token IDs for BERT
        attention_mask -> tells BERT which tokens are real
        token_type_ids -> segment IDs, included for compatibility
        metadata       -> numerical metadata vector
        labels         -> correct class label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        metadata_matrix: np.ndarray,
        text_column: str,
        label_column: str,
        max_length: int,
    ):
        # Store all statement texts as strings. Missing text becomes "".
        self.texts    = df[text_column].fillna("").astype(str).tolist()
        # Store class labels as integers.
        self.labels   = df[label_column].astype(int).tolist()
        # Convert preprocessed metadata matrix from numpy to a float tensor.
        # Metadata must be float because it will be concatenated with BERT embeddings.
        self.metadata = torch.tensor(metadata_matrix, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts) # Nr of examples in the dataset

    def __getitem__(self, idx):
        # Tokenize one statement for BERT.
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Return both the BERT text inputs + the metadata vector.
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "metadata":       self.metadata[idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3.  Fusion model
# ---------------------------------------------------------------------------

class BertMetadataFusion(nn.Module):
    """
    The neural network used for BERT + metadata fusion.

    This class is the core of the metadata model. It explicitly shows how the
    text representation and metadata representation are joined.

    BERT encoder  +  metadata MLP  →  fusion head  →  classifier.

    Text path  : BERT [CLS] token  →  768-dim vector
    Metadata path: flat sklearn-encoded vector (no gradient)
    Fusion     : concat  →  Linear(768 + meta_dim, HIDDEN_DIM)  →  ReLU
                          →  Dropout  →  Linear(HIDDEN_DIM, num_labels)
    """

    def __init__(self, bert_model_name: str, meta_dim: int, num_labels: int):
        super().__init__()
        # Load pretrained BERT encoder without a built-in classification head.
        # We build a custom fusion head 
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # For bert-base-uncased, each token representation has 768 dimensions.
        bert_dim  = self.bert.config.hidden_size          # 768 for bert-base

        # The fusion head receives both:
        #   - BERT [CLS] vector: 768 numbers
        #   - metadata vector: meta_dim numbers
        # and outputs one logit score per class.
        self.fusion_head = nn.Sequential(
            nn.Linear(bert_dim + meta_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        # Forward pass through BERT.
        # last_hidden_state shape: (batch_size, sequence_length, 768).
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] representation - shape (batch, 768).
        # Index 0 is the special [CLS] token, commonly used as a summary of the text.
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Concatenate with metadata - shape (batch, 768 + meta_dim).
        # dim=-1 means concatenate along the feature dimension, not across examples.
        # This is the exact fusion step: text information + metadata information.
        fused = torch.cat([cls_embedding, metadata], dim=-1)

        # Return raw class scores, called logits. CrossEntropyLoss expects logits.
        return self.fusion_head(fused)          # (batch, num_labels)


# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------

# One epoch means one complete pass through the training set.
def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    # Enable training mode: dropout is active and gradients are tracked.
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Each batch contains BERT inputs, metadata vectors, and labels.
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        metadata       = batch["metadata"].to(device)
        labels         = batch["labels"].to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()
        # Forward pass through the fusion model.
        logits = model(input_ids, attention_mask, token_type_ids, metadata)

        # Compare predicted logits with true labels.
        loss   = criterion(logits, labels)
        # Backpropagation: compute gradients for BERT and the fusion head.
        # The sklearn metadata preprocessing itself is not trained here.
        loss.backward()

        # Gradient clipping - standard practice for fine-tuning BERT
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameters.
        optimizer.step()

        # Update the learning rate according to the warm-up/decay schedule.
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# During evaluation, gradients are not needed, saving memory and computation.
@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    # Evaluation mode disables dropout.
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        metadata       = batch["metadata"].to(device)
        labels         = batch["labels"]

        # Forward pass on the test batch.
        logits = model(input_ids, attention_mask, token_type_ids, metadata)

        # Choose the class with the highest logit score.
        preds  = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# 5.  Main experiment runner
# ---------------------------------------------------------------------------

# Prefer raw statements when available, otherwise use cleaned text.
def _pick_text_column(df: pd.DataFrame) -> str:
    return (
        TEXT_COLUMN_PREFERRED
        if TEXT_COLUMN_PREFERRED in df.columns
        else TEXT_COLUMN_FALLBACK
    )


# Runs one full experiment: load data, preprocess metadata, train, evaluate, save.
def run_experiment(name: str, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # --- Load data ---
    # Training uses train + validation combined. The test set is kept separate.
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

    # --- Fit metadata transformers on training data only ---
    # This is important to avoid data leakage. The test set must be transformed
    # using encoders/scalers learned only from the training data.
    meta_transformers = MetadataTransformers()
    train_meta = meta_transformers.fit(train_df)
    test_meta  = meta_transformers.transform(test_df)
    # meta_dim is the length of the final metadata vector after all one-hot,
    # numeric, and subject features have been concatenated.
    meta_dim   = train_meta.shape[1]
    print(f"  Metadata dim: {meta_dim}")

    # --- Tokenizer ---
    # Converts raw statement text into BERT token IDs.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FusionDataset(
        df=train_df,
        tokenizer=tokenizer,
        metadata_matrix=train_meta,
        text_column=text_column,
        label_column=label_column,
        max_length=MAX_LENGTH,
    )
    test_dataset = FusionDataset(
        df=test_df,
        tokenizer=tokenizer,
        metadata_matrix=test_meta,
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
    # Use GPU if available; otherwise CPU works but is much slower for BERT.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Model ---
    # Create a fresh fusion model for this experiment. The multiclass and binary
    # experiments are trained separately because they have different output sizes.
    model = BertMetadataFusion(
        bert_model_name=MODEL_NAME,
        meta_dim=meta_dim,
        num_labels=num_labels,
    ).to(device)

    # --- Optimiser & scheduler ---
    # AdamW is standard for fine-tuning transformer models. The scheduler warms
    # up the learning rate and then decays it linearly.
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

# CrossEntropyLoss is used for classification with integer labels.
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={avg_loss:.4f}")

    # --- Final evaluation on held-out test set ---
    # The test set is used only here, after training is complete.
    predictions, y_true = predict(model, test_loader, device)

    # Save all main evaluation outputs in a dictionary.
    # This dictionary is written to JSON and summarized in TXT.
    metrics = {
        "dataset":      name,
        "model_name":   MODEL_NAME,
        "text_column":  text_column,
        "label_column": label_column,
        "num_labels":   int(num_labels),
        "meta_dim":     int(meta_dim),
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


# Save both a detailed JSON file and a compact human-readable TXT summary.
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
        f"meta_dim: {metrics['meta_dim']}",
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

# Entry point. This runs all experiments listed in EXPERIMENTS.
def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {MODEL_NAME}")
    print(f"Data dir   : {CLEANED_TEXT_DIR}")
    print(f"Metrics dir: {METRICS_DIR}")
    print(f"Models dir : {MODELS_DIR}")

    if torch.cuda.is_available():
        print(f"CUDA: yes  -  GPU: {torch.cuda.get_device_name(0)}")
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