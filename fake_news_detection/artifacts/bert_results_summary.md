# BERT-Based Fake News Classification Models: Architecture and Results

## 1. Overview

This report summarizes and compares two BERT-based models implemented for fake news classification:

1. **BERT text-only baseline**
2. **BERT + metadata fusion model**

Both models were implemented using a custom PyTorch training loop and evaluated on the same held-out test splits. The purpose of the comparison is to determine whether adding structured metadata improves classification performance compared with using only the statement text.

The main conclusion is that **the BERT + metadata model outperforms the text-only baseline on both the binary and multiclass classification tasks**, with the multiclass gap being more substantial. The improvement is driven by large gains on minority classes (notably class 4 and class 5) that the text-only model struggles to predict.

---

## 2. Experimental Goal

The experiment compares two modeling strategies:

- using only the textual content of a claim or statement;
- using the textual content together with additional metadata such as speaker, party, state, subject, context, and historical count features.

The comparison is controlled because both models use the same core BERT encoder and the same training configuration. Therefore, the main experimental difference is whether metadata is added to the BERT text representation.

---

## 3. Shared Experimental Setup

Both models use the following setup:

| Component                 | Configuration                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------- |
| Pretrained language model | `bert-base-uncased`                                                                    |
| Text column               | `statement`                                                                            |
| Maximum sequence length   | `128`                                                                                  |
| Batch size                | `16`                                                                                   |
| Number of epochs          | `3`                                                                                    |
| Learning rate             | Selected via search over {`1e-5`, `2e-5`, `5e-5`}; best chosen by final-epoch val loss |
| Optimizer                 | `AdamW`                                                                                |
| Scheduler                 | Linear warm-up scheduler                                                               |
| Training data             | Train split only (validation split held out for per-epoch monitoring)                  |
| Evaluation data           | Held-out test split                                                                    |
| Phase 1 — frozen          | 1 epoch: BERT encoder frozen, head trained only                                        |
| Phase 2 — unfrozen        | 2 epochs: all layers fine-tuned at the searched LR                                     |
| Evaluation metrics        | Accuracy, macro F1, weighted F1, classification report, confusion matrix               |

Two classification tasks were evaluated:

| Task                      | Training File                | Validation File              | Test File                   | Label Column | Number of Classes |
| ------------------------- | ---------------------------- | ---------------------------- | --------------------------- | ------------ | ----------------: |
| Multiclass classification | `train.processed.csv`        | `valid.processed.csv`        | `test.processed.csv`        | `label6_int` |                 6 |
| Binary classification     | `train_binary.processed.csv` | `valid_binary.processed.csv` | `test_binary.processed.csv` | `label2_int` |                 2 |

---

## 4. Model Architectures

### 4.1 Model 1: BERT Text-Only Baseline

The first model uses only the statement text. The text is tokenized using the BERT tokenizer and passed through `bert-base-uncased`. The model then extracts the `[CLS]` embedding, which acts as the sentence-level representation of the statement.

#### Architecture Flow

```text
statement text
    ↓
BERT tokenizer
    ↓
BERT encoder
    ↓
[CLS] embedding
    ↓
classifier head
    ↓
predicted label
```

#### Classifier Head

```text
Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → num_labels)
```

The value of `num_labels` depends on the task:

- `num_labels = 6` for multiclass classification;
- `num_labels = 2` for binary classification.

This model is the baseline because it relies only on the semantic information contained in the statement text.

---

### 4.2 Model 2: BERT + Metadata Fusion

The second model extends the text-only BERT baseline by adding a metadata branch. It still uses the BERT `[CLS]` embedding as the text representation, but this representation is concatenated with a processed metadata vector before classification.

#### Architecture Flow

```text
statement text
    ↓
BERT tokenizer
    ↓
BERT encoder
    ↓
[CLS] embedding

metadata columns
    ↓
metadata preprocessing
    ↓
metadata vector

[CLS] embedding + metadata vector
    ↓
fusion classifier head
    ↓
predicted label
```

#### Metadata Features

The metadata model uses three types of metadata features.

| Feature Type         | Columns                                               | Processing Method                                      |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| Categorical metadata | `speaker`, `party`, `state`, `speaker_job`, `context` | Label encoding followed by one-hot encoding            |
| Numeric metadata     | `hist1`, `hist2`, `hist3`, `hist4`, `hist5`           | Standardization with `StandardScaler`                  |
| Multi-value metadata | `subjects`                                            | Split by comma and encoded using `MultiLabelBinarizer` |

After preprocessing, all metadata features are concatenated into a single metadata vector.

#### Fusion Classifier Head

```text
Linear(768 + metadata_dim → 256) → ReLU → Dropout(0.3) → Linear(256 → num_labels)
```

The metadata dimension differs by task because the binary and multiclass datasets contain different encoded metadata vocabularies.

| Task                       | Metadata Dimension |
| -------------------------- | -----------------: |
| Binary BERT + metadata     |               6424 |
| Multiclass BERT + metadata |               8718 |

The large size of the metadata vector is important when interpreting the results, because high-dimensional sparse metadata can introduce noise or make optimization more difficult.

---

## 5. Output Artifacts

### 5.1 BERT Text-Only Outputs

Running the text-only script produces the following result and model artifacts:

```text
artifacts/final/bert_text_only/
    multiclass_bert_text_only_metrics.json
    multiclass_bert_text_only_summary.txt
    binary_bert_text_only_metrics.json
    binary_bert_text_only_summary.txt

artifacts/models/bert_text_only/
    multiclass_bert_text_only/
        model_weights.pt
        tokenizer files
    binary_bert_text_only/
        model_weights.pt
        tokenizer files
```

### 5.2 BERT + Metadata Outputs

Running the metadata-fusion script produces the following result and model artifacts:

```text
artifacts/final/bert_text_metadata/
    multiclass_bert_metadata_metrics.json
    multiclass_bert_metadata_summary.txt
    binary_bert_metadata_metrics.json
    binary_bert_metadata_summary.txt

artifacts/models/bert_text_metadata/
    multiclass_bert_metadata/
        model_weights.pt
        tokenizer files
    binary_bert_metadata/
        model_weights.pt
        tokenizer files
```

### 5.3 Metrics Saved by Both Models

Each metrics JSON file contains:

- dataset name;
- model name;
- text column used;
- label column used;
- number of labels;
- maximum sequence length;
- number of training rows;
- number of evaluation rows;
- accuracy;
- macro F1-score;
- weighted F1-score;
- labels present in the test set;
- confusion matrix;
- full classification report.

---

## 6. Overall Results

### 6.1 Binary Classification Results

| Model           | Train Rows | Test Rows | Accuracy | Macro F1 | Weighted F1 |
| --------------- | ---------: | --------: | -------: | -------: | ----------: |
| BERT text-only  |       6489 |       802 |   0.6845 |   0.6664 |      0.6778 |
| BERT + metadata |       6489 |       802 |   0.6858 |   0.6695 |      0.6803 |

#### Binary Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric      | Difference |
| ----------- | ---------: |
| Accuracy    |    +0.0013 |
| Macro F1    |    +0.0031 |
| Weighted F1 |    +0.0025 |

#### Interpretation

For the binary task, the metadata model performs marginally better on all three metrics. The gains are small (under 0.3 percentage points), so the two models are essentially equivalent on the binary task. The metadata adds a very slight edge.

---

### 6.2 Multiclass Classification Results

| Model           | Train Rows | Test Rows | Accuracy | Macro F1 | Weighted F1 |
| --------------- | ---------: | --------: | -------: | -------: | ----------: |
| BERT text-only  |      10269 |      1283 |   0.2642 |   0.2277 |      0.2500 |
| BERT + metadata |      10269 |      1283 |   0.2783 |   0.2631 |      0.2677 |

#### Multiclass Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric      | Difference |
| ----------- | ---------: |
| Accuracy    |    +0.0141 |
| Macro F1    |    +0.0354 |
| Weighted F1 |    +0.0177 |

#### Interpretation

For the multiclass task, the metadata model clearly outperforms the text-only baseline. It improves accuracy by 1.41 percentage points and macro F1 by 3.54 percentage points. The macro F1 improvement is especially meaningful because it is averaged equally across all six classes, including the minority classes where the text-only model struggles most.

The multiclass task is more difficult than the binary task because the model must distinguish between six fine-grained truthfulness categories rather than two broader classes.

---

## 7. Class-Level Analysis

### 7.1 Binary Classification: Class-Level Results

| Class | Text-only F1 | Metadata F1 | F1 Difference | Text-only Recall | Metadata Recall | Recall Difference |
| ----- | -----------: | ----------: | ------------: | ---------------: | --------------: | ----------------: |
| 0     |       0.5886 |      0.5962 |       +0.0076 |           0.5292 |          0.5439 |           +0.0147 |
| 1     |       0.7442 |      0.7429 |       -0.0013 |           0.8000 |          0.7913 |           -0.0087 |

#### Binary Confusion Matrices

BERT text-only:

```text
[[181, 161],
 [ 92, 368]]
```

BERT + metadata:

```text
[[186, 156],
 [ 96, 364]]
```

#### Binary Class-Level Interpretation

The metadata model correctly predicts slightly more examples from class `0`:

```text
Text-only class 0 correct predictions: 181
Metadata class 0 correct predictions: 186
```

The text-only model correctly predicts slightly more examples from class `1`:

```text
Text-only class 1 correct predictions: 368
Metadata class 1 correct predictions: 364
```

The metadata model shifts a small number of predictions toward class `0`, improving recall for class `0` (+1.47 pp) while slightly reducing recall for class `1` (-0.87 pp). The net effect on macro F1 is a marginal improvement (+0.0031). Both models are essentially equivalent on the binary task.

---

### 7.2 Multiclass Classification: Class-Level Results

| Class | Text-only F1 | Metadata F1 | F1 Difference | Text-only Recall | Metadata Recall | Recall Difference |
| ----- | -----------: | ----------: | ------------: | ---------------: | --------------: | ----------------: |
| 0     |       0.1953 |      0.1290 |       -0.0663 |           0.1542 |          0.0841 |           -0.0701 |
| 1     |       0.3139 |      0.3114 |       -0.0025 |           0.3880 |          0.3400 |           -0.0480 |
| 2     |       0.2747 |      0.2827 |       +0.0080 |           0.3071 |          0.3483 |           +0.0412 |
| 3     |       0.2995 |      0.3058 |       +0.0063 |           0.3494 |          0.3414 |           -0.0080 |
| 4     |       0.0600 |      0.2481 |       +0.1881 |           0.0326 |          0.1739 |           +0.1413 |
| 5     |       0.2229 |      0.3015 |       +0.0786 |           0.1754 |          0.2844 |           +0.1090 |

#### Multiclass Class-Level Interpretation

The metadata model substantially improves performance on minority classes:

| Improved Class | F1 Improvement | Recall Improvement |
| -------------- | -------------: | -----------------: |
| Class 2        |        +0.0080 |            +0.0412 |
| Class 3        |        +0.0063 |            -0.0080 |
| Class 4        |        +0.1881 |            +0.1413 |
| Class 5        |        +0.0786 |            +0.1090 |

It performs worse on the two most frequent classes:

| Weaker Class | F1 Change | Recall Change |
| ------------ | --------: | ------------: |
| Class 0      |   -0.0663 |       -0.0701 |
| Class 1      |   -0.0025 |       -0.0480 |

The most striking result is **class 4**: the text-only model nearly ignores it (recall 0.0326, F1 0.0600), while the metadata model achieves a recall of 0.1739 and F1 of 0.2481 — an improvement of +0.1881 F1. Class 5 also improves substantially (+0.0786 F1, +0.1090 recall). These large gains on minority classes are what drives the overall macro F1 improvement (+0.0354), outweighing the drops on classes 0 and 1.

---

## 8. Comparison Summary

| Task                      | Better Model                 | Main Reason                                                                                       |
| ------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------- |
| Binary classification     | BERT + metadata (marginally) | Slightly higher accuracy, macro F1, and weighted F1                                               |
| Multiclass classification | BERT + metadata              | Clear improvement in accuracy (+1.41 pp) and macro F1 (+3.54 pp); large gains on minority classes |

Across both tasks, the metadata-fusion model is the stronger model in this experiment. The advantage is very small for binary classification but meaningful for multiclass classification.

The result suggests that **structured metadata is a valuable complement to BERT text representations**, especially for fine-grained truthfulness classification. Speaker identity, venue, subject, and context provide signal that cannot be recovered from statement text alone.

---

## 9. Notes on Metadata Representation

The metadata-fusion model adds thousands of additional features:

- **6424 metadata features** for the binary task;
- **8718 metadata features** for the multiclass task.

Despite the large feature space, the metadata model outperforms the text-only baseline. However, several architectural challenges remain relevant for future improvement.

### 9.1 High Dimensionality

One-hot and multi-hot metadata encoding creates a very large sparse vector. Many metadata values appear rarely. The current approach works but could be made more efficient with dimensionality reduction or learned categorical embeddings.

### 9.2 Sparsity

Most entries in the metadata vector are zero for any given example. Despite this sparsity, the metadata model still improves overall — largely because rare-but-informative fields (such as speaker identity) provide useful signal for minority classes.

### 9.3 Noise in Metadata

Some metadata fields may not be strongly predictive of the truthfulness label. This likely explains why the metadata model underperforms on classes 0 and 1 even as it substantially improves minority classes 4 and 5.

### 9.4 Simple Fusion Strategy

The current model concatenates the BERT `[CLS]` embedding directly with the metadata vector. A separate metadata projection layer, attention-based fusion, or cross-modal gating mechanism may extract more structured signal and further improve performance.

### 9.5 Limited Dataset Size

The metadata branch has thousands of input features, but the dataset contains only thousands of training examples. This increases overfitting risk for rare speakers, subjects, states, or contexts. Stronger regularization or selective feature selection could help.

---

## 10. Final Conclusion

Two BERT-based fake news classification models were implemented and evaluated: a text-only BERT baseline and a BERT + metadata fusion model. The text-only model tokenizes each statement, passes it through `bert-base-uncased`, extracts the `[CLS]` embedding, and feeds it into a classifier head. The metadata model extends this architecture by preprocessing structured metadata and concatenating the resulting metadata vector with the BERT `[CLS]` embedding before classification.

Both models were trained using a two-phase strategy: BERT encoder weights are frozen for one epoch (head-only training), then unfrozen for two additional epochs of full fine-tuning. The learning rate is selected per experiment via a search over `{1e-5, 2e-5, 5e-5}`, and a held-out validation split is used for per-epoch monitoring throughout.

The comparison shows that the **BERT + metadata model outperforms the text-only baseline on both tasks**. For binary classification, the metadata model achieves **0.6858 accuracy** and **0.6695 macro F1**, compared with **0.6845 accuracy** and **0.6664 macro F1** for the text-only model — a marginal difference. For multiclass classification, the metadata model achieves **0.2783 accuracy** and **0.2631 macro F1**, compared with **0.2642 accuracy** and **0.2277 macro F1** for the text-only model — a meaningful improvement of 1.41 pp accuracy and 3.54 pp macro F1.

The multiclass gains are largely explained by large improvements on minority classes 4 and 5, which the text-only model nearly fails to detect. Metadata features encoding speaker identity, subject, context, and venue provide complementary signal that helps the model distinguish fine-grained truthfulness categories. The metadata-fusion architecture is therefore the stronger model, particularly for the harder multiclass task.
