# BERT-Based Fake News Classification Models: Architecture and Results

## 1. Overview

This report summarizes and compares two BERT-based models implemented for fake news classification:

1. **BERT text-only baseline**
2. **BERT + metadata fusion model**

Both models were implemented using a custom PyTorch training loop and evaluated on the same held-out test splits. The purpose of the comparison is to determine whether adding structured metadata improves classification performance compared with using only the statement text.

The main conclusion is that **the BERT text-only model performs slightly better than the BERT + metadata model on both the binary and multiclass classification tasks**. Although the metadata model improves some individual classes, its overall accuracy, macro F1-score, and weighted F1-score are lower than the text-only baseline.

---

## 2. Experimental Goal

The experiment compares two modeling strategies:

- using only the textual content of a claim or statement;
- using the textual content together with additional metadata such as speaker, party, state, subject, context, and historical count features.

The comparison is controlled because both models use the same core BERT encoder and the same training configuration. Therefore, the main experimental difference is whether metadata is added to the BERT text representation.

---

## 3. Shared Experimental Setup

Both models use the following setup:

| Component | Configuration |
|---|---|
| Pretrained language model | `bert-base-uncased` |
| Text column | `statement` |
| Maximum sequence length | `128` |
| Batch size | `16` |
| Number of epochs | `3` |
| Learning rate | `2e-5` |
| Optimizer | `AdamW` |
| Scheduler | Linear warm-up scheduler |
| Training data | Train + validation split |
| Evaluation data | Held-out test split |
| Evaluation metrics | Accuracy, macro F1, weighted F1, classification report, confusion matrix |

Two classification tasks were evaluated:

| Task | Training Files | Test File | Label Column | Number of Classes |
|---|---|---|---|---:|
| Multiclass classification | `train.processed.csv`, `valid.processed.csv` | `test.processed.csv` | `label6_int` | 6 |
| Binary classification | `train_binary.processed.csv`, `valid_binary.processed.csv` | `test_binary.processed.csv` | `label2_int` | 2 |

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

| Feature Type | Columns | Processing Method |
|---|---|---|
| Categorical metadata | `speaker`, `party`, `state`, `speaker_job`, `context` | Label encoding followed by one-hot encoding |
| Numeric metadata | `hist1`, `hist2`, `hist3`, `hist4`, `hist5` | Standardization with `StandardScaler` |
| Multi-value metadata | `subjects` | Split by comma and encoded using `MultiLabelBinarizer` |

After preprocessing, all metadata features are concatenated into a single metadata vector.

#### Fusion Classifier Head

```text
Linear(768 + metadata_dim → 256) → ReLU → Dropout(0.3) → Linear(256 → num_labels)
```

The metadata dimension differs by task because the binary and multiclass datasets contain different encoded metadata vocabularies.

| Task | Metadata Dimension |
|---|---:|
| Binary BERT + metadata | 6961 |
| Multiclass BERT + metadata | 9441 |

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

| Model | Train Rows | Test Rows | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BERT text-only | 7288 | 802 | 0.6908 | 0.6769 | 0.6868 |
| BERT + metadata | 7288 | 802 | 0.6796 | 0.6581 | 0.6707 |

#### Binary Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric | Difference |
|---|---:|
| Accuracy | -0.0112 |
| Macro F1 | -0.0189 |
| Weighted F1 | -0.0161 |

#### Interpretation

For the binary task, the text-only BERT model performs better overall. It achieves higher accuracy, macro F1, and weighted F1. The metadata-fusion model performs worse by approximately:

- **1.12 percentage points** in accuracy;
- **1.89 percentage points** in macro F1;
- **1.61 percentage points** in weighted F1.

This suggests that adding metadata did not improve binary fake news classification in the current setup.

---

### 6.2 Multiclass Classification Results

| Model | Train Rows | Test Rows | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BERT text-only | 11553 | 1283 | 0.2845 | 0.2724 | 0.2814 |
| BERT + metadata | 11553 | 1283 | 0.2806 | 0.2705 | 0.2756 |

#### Multiclass Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric | Difference |
|---|---:|
| Accuracy | -0.0039 |
| Macro F1 | -0.0018 |
| Weighted F1 | -0.0057 |

#### Interpretation

For the multiclass task, the text-only model also performs slightly better overall. The differences are smaller than in the binary task, but the metadata-fusion model is still lower on all three main metrics.

The multiclass task is more difficult than the binary task because the model must distinguish between six fine-grained truthfulness categories rather than two broader classes.

---

## 7. Class-Level Analysis

### 7.1 Binary Classification: Class-Level Results

| Class | Text-only F1 | Metadata F1 | F1 Difference | Text-only Recall | Metadata Recall | Recall Difference |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.6101 | 0.5724 | -0.0377 | 0.5673 | 0.5029 | -0.0643 |
| 1 | 0.7438 | 0.7438 | -0.0000 | 0.7826 | 0.8109 | +0.0283 |

#### Binary Confusion Matrices

BERT text-only:

```text
[[194, 148],
 [100, 360]]
```

BERT + metadata:

```text
[[172, 170],
 [ 87, 373]]
```

#### Binary Class-Level Interpretation

The text-only model correctly predicts more examples from class `0`:

```text
Text-only class 0 correct predictions: 194
Metadata class 0 correct predictions: 172
```

The metadata model correctly predicts more examples from class `1`:

```text
Text-only class 1 correct predictions: 360
Metadata class 1 correct predictions: 373
```

However, the metadata model also misclassifies more class `0` examples as class `1`. This hurts macro F1 and overall accuracy. In other words, the metadata model appears to shift predictions toward class `1`, which improves recall for class `1` but weakens performance on class `0`.

---

### 7.2 Multiclass Classification: Class-Level Results

| Class | Text-only F1 | Metadata F1 | F1 Difference | Text-only Recall | Metadata Recall | Recall Difference |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2454 | 0.1956 | -0.0498 | 0.2196 | 0.1449 | -0.0748 |
| 1 | 0.3096 | 0.3168 | +0.0072 | 0.2960 | 0.3200 | +0.0240 |
| 2 | 0.2809 | 0.2500 | -0.0309 | 0.3109 | 0.2884 | -0.0225 |
| 3 | 0.3108 | 0.3264 | +0.0156 | 0.3695 | 0.3775 | +0.0080 |
| 4 | 0.2031 | 0.2370 | +0.0339 | 0.1413 | 0.1739 | +0.0326 |
| 5 | 0.2843 | 0.2974 | +0.0131 | 0.2654 | 0.2938 | +0.0284 |


#### Multiclass Class-Level Interpretation

The metadata model improves performance for some classes:

| Improved Class | F1 Improvement | Recall Improvement |
|---|---:|---:|
| Class 1 | +0.0072 | +0.0240 |
| Class 3 | +0.0156 | +0.0080 |
| Class 4 | +0.0339 | +0.0326 |
| Class 5 | +0.0131 | +0.0284 |

However, it performs worse on other classes:

| Weaker Class | F1 Change | Recall Change |
|---|---:|---:|
| Class 0 | -0.0498 | -0.0748 |
| Class 2 | -0.0309 | -0.0225 |

This explains why the metadata model does not improve the overall multiclass result. Its gains are not consistent across all classes, and the drop for classes `0` and `2` offsets the improvements for classes `1`, `3`, `4`, and `5`.

---

## 8. Comparison Summary

| Task | Better Model | Main Reason |
|---|---|---|
| Binary classification | BERT text-only | Higher accuracy, macro F1, and weighted F1 |
| Multiclass classification | BERT text-only | Slightly higher accuracy, macro F1, and weighted F1 |

Across both tasks, the text-only model is the stronger model in this experiment.

The result does not necessarily mean that metadata is useless. It means that **the current metadata encoding and fusion strategy did not improve performance**. The metadata representation may be too sparse, too noisy, or too high-dimensional for the available amount of training data.

---

## 9. Why the Metadata Model May Underperform

The metadata-fusion model adds thousands of additional features:

- **6961 metadata features** for the binary task;
- **9441 metadata features** for the multiclass task.

This can make the model harder to train for several reasons.

### 9.1 High Dimensionality

One-hot and multi-hot metadata encoding creates a very large sparse vector. Many metadata values may appear only rarely, making it difficult for the model to learn reliable patterns.

### 9.2 Sparsity

Most entries in the metadata vector are likely zero for any given example. Sparse metadata can be useful, but it can also increase variance and make the classifier more sensitive to rare categories.

### 9.3 Noise in Metadata

Some metadata fields may not be strongly predictive of the truthfulness label. If noisy features are concatenated with the BERT representation, the classifier may learn less stable decision boundaries.

### 9.4 Simple Fusion Strategy

The current model uses direct concatenation of the BERT `[CLS]` embedding and the metadata vector. This is a reasonable baseline, but it may not be the best way to combine text and metadata. A separate metadata projection layer or attention-based fusion mechanism may work better.

### 9.5 Limited Dataset Size

The metadata branch has thousands of input features, but the dataset contains only thousands of training examples. This can increase the risk of overfitting, especially for rare speakers, subjects, states, or contexts.

---

## 10. Final Conclusion

Two BERT-based fake news classification models were implemented and evaluated: a text-only BERT baseline and a BERT + metadata fusion model. The text-only model tokenizes each statement, passes it through `bert-base-uncased`, extracts the `[CLS]` embedding, and feeds it into a classifier head. The metadata model extends this architecture by preprocessing structured metadata and concatenating the resulting metadata vector with the BERT `[CLS]` embedding before classification.

The comparison shows that the **BERT text-only model outperforms the BERT + metadata model on both tasks**. For binary classification, the text-only model achieves **0.6908 accuracy** and **0.6769 macro F1**, compared with **0.6796 accuracy** and **0.6581 macro F1** for the metadata model. For multiclass classification, the text-only model achieves **0.2845 accuracy** and **0.2724 macro F1**, compared with **0.2806 accuracy** and **0.2705 macro F1** for the metadata model.

Therefore, the text-only BERT model is the stronger model in the current experiment. The results suggest that adding metadata through direct high-dimensional concatenation does not improve performance. A more effective metadata architecture may require dimensionality reduction, learned categorical embeddings, stronger regularization, or a more selective use of metadata features.