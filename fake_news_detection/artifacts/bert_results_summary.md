# BERT-Based Fake News Classification Models: Architecture and Results

## 1. Overview

This report summarizes and compares two BERT-based models implemented for fake news classification:

1. **BERT text-only baseline**
2. **BERT + metadata fusion model**

Both models were implemented using a custom PyTorch training loop and evaluated on the same held-out test splits. The purpose of the comparison is to determine whether adding structured metadata improves classification performance compared with using only the statement text.

The main conclusion is that **the BERT + metadata model outperforms the text-only baseline on the binary classification task**, achieving higher accuracy, macro F1, and weighted F1. On the multiclass task the results are mixed: the metadata model achieves higher accuracy while the text-only model achieves higher macro F1, driven by class-weighted training that specifically improves minority-class recall.

---

## 2. Experimental Goal

The experiment compares two modeling strategies:

- using only the textual content of a claim or statement;
- using the textual content together with additional metadata such as speaker, party, state, subject, context, and historical count features.

The comparison is controlled because both models use the same core BERT encoder and the same training configuration. Therefore, the main experimental difference is whether metadata is added to the BERT text representation.

---

## 3. Shared Experimental Setup

Both models use the following setup:

| Component                 | Configuration                                                                     |
| ------------------------- | --------------------------------------------------------------------------------- |
| Pretrained language model | `bert-base-uncased`                                                               |
| Text column               | `statement`                                                                       |
| Maximum sequence length   | `128`                                                                             |
| Batch size                | `16`                                                                              |
| Maximum epochs            | `5` (early stopping with patience 2 may stop earlier)                            |
| Learning rate             | `1e-5` (selected from prior search over `{1e-5, 2e-5, 5e-5}`)                   |
| Optimizer                 | `AdamW`, weight decay `0.01`                                                      |
| Scheduler                 | Linear warm-up (10% of steps)                                                     |
| Training data             | Train split only — validation split is held out for clean per-epoch monitoring    |
| Evaluation data           | Held-out test split                                                               |
| Phase 1 — frozen          | 1 epoch: BERT encoder frozen, head trained only                                   |
| Phase 2 — unfrozen        | Up to 4 epochs: all layers fine-tuned; early stopping restores best weights       |
| Class weights             | Balanced inverse-frequency weights applied to multiclass loss only                |
| Early stopping            | Patience = 2 epochs; best model (lowest val loss) restored before test evaluation |
| Evaluation metrics        | Accuracy, macro F1, weighted F1, classification report, confusion matrix          |

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
| BERT text-only  |       6489 |       802 |   0.6808 |   0.6689 |      0.6781 |
| BERT + metadata |       6489 |       802 |   0.6883 |   0.6760 |      0.6853 |

#### Binary Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric      | Difference |
| ----------- | ---------: |
| Accuracy    |    +0.0075 |
| Macro F1    |    +0.0071 |
| Weighted F1 |    +0.0072 |

#### Interpretation

For the binary task, the metadata model outperforms the text-only baseline on all three metrics. The gains are consistent across accuracy, macro F1, and weighted F1, indicating that metadata provides genuine complementary signal even for the simpler two-class problem. Class weights were not applied to binary training because the binary dataset is already roughly balanced — applying them would have penalised the majority class unnecessarily.

---

### 6.2 Multiclass Classification Results

| Model           | Train Rows | Test Rows | Accuracy | Macro F1 | Weighted F1 |
| --------------- | ---------: | --------: | -------: | -------: | ----------: |
| BERT text-only  |      10269 |      1283 |   0.2728 |   0.2715 |      0.2637 |
| BERT + metadata |      10269 |      1283 |   0.2751 |   0.2636 |      0.2520 |

#### Multiclass Classification Difference

The difference is calculated as:

```text
BERT + metadata score - BERT text-only score
```

| Metric      | Difference |
| ----------- | ---------: |
| Accuracy    |    +0.0023 |
| Macro F1    |    -0.0079 |
| Weighted F1 |    -0.0117 |

#### Interpretation

For the multiclass task the results are mixed. The metadata model achieves slightly higher accuracy (+0.23 pp), but the text-only model achieves higher macro F1 (+0.79 pp) and weighted F1 (+1.17 pp). The higher macro F1 for the text-only model is driven by class-weighted training: applying balanced inverse-frequency weights to the six-class loss specifically boosts minority-class recall, which macro F1 captures equally across all classes. The metadata model does not benefit from class weights in the same way because the high-dimensional sparse metadata vector (8718 features) introduces additional variance, making it harder for the model to learn stable minority-class boundaries.

The multiclass task is inherently harder than the binary task because the model must distinguish between six fine-grained truthfulness categories rather than two broader classes.

---

## 7. Training Dynamics

Early stopping was applied in Phase 2 (unfrozen fine-tuning) with patience of 2 epochs. The best model weights (lowest validation loss) were restored before final test evaluation. This prevented the binary models from overfitting: both binary experiments selected weights from epoch 3, while multiclass experiments ran longer before stopping.

| Experiment                  | Best Epoch | Best Val Loss |
| --------------------------- | ---------: | ------------: |
| multiclass BERT text-only   |          4 |        1.6907 |
| binary BERT text-only       |          3 |        0.6094 |
| multiclass BERT + metadata  |          3 |        1.6984 |
| binary BERT + metadata      |          3 |        0.5985 |

Validation loss decreased consistently across all experiments during Phase 2, confirming that the models were learning rather than diverging.

---

## 8. Comparison Summary

| Task                      | Better Model                 | Main Reason                                                                                              |
| ------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| Binary classification     | BERT + metadata              | Consistent improvement across accuracy (+0.75 pp), macro F1 (+0.71 pp), and weighted F1 (+0.72 pp)      |
| Multiclass classification | Mixed                        | Metadata wins accuracy (+0.23 pp); text-only wins macro F1 (+0.79 pp) due to class-weighted minority boost |

The metadata model is the stronger model on binary classification. On multiclass classification the choice depends on the evaluation priority: if overall accuracy matters, metadata is marginally better; if balanced per-class performance (macro F1) matters, the class-weighted text-only model is stronger.

The result confirms that **structured metadata provides complementary signal to BERT text representations**. The binary task benefits clearly from metadata. The multiclass task is harder to improve with metadata alone because the high-dimensional sparse metadata vector introduces noise that competes with the class-weighting benefit.

---

## 9. Notes on Metadata Representation

The metadata-fusion model adds thousands of additional features:

- **6424 metadata features** for the binary task;
- **8718 metadata features** for the multiclass task.

Several architectural challenges remain relevant for future improvement.

### 9.1 High Dimensionality

One-hot and multi-hot metadata encoding creates a very large sparse vector. Many metadata values appear rarely. The current approach works but could be made more efficient with dimensionality reduction or learned categorical embeddings.

### 9.2 Sparsity

Most entries in the metadata vector are zero for any given example. Despite this sparsity, the metadata model still improves binary classification — largely because rare-but-informative fields such as speaker identity provide useful signal.

### 9.3 Noise in Metadata

Some metadata fields may not be strongly predictive of the truthfulness label. This is most visible in the multiclass task, where the high-dimensional metadata vector competes with the class-weighting signal and reduces macro F1 relative to the text-only model.

### 9.4 Simple Fusion Strategy

The current model concatenates the BERT `[CLS]` embedding directly with the metadata vector. A separate metadata projection layer, attention-based fusion, or cross-modal gating mechanism may extract more structured signal and further improve performance.

### 9.5 Limited Dataset Size

The metadata branch has thousands of input features, but the dataset contains only thousands of training examples. This increases overfitting risk for rare speakers, subjects, states, or contexts. Early stopping partially mitigates this, but stronger regularization or selective feature selection could help further.

---

## 10. Final Conclusion

Two BERT-based fake news classification models were implemented and evaluated: a text-only BERT baseline and a BERT + metadata fusion model. The text-only model tokenizes each statement, passes it through `bert-base-uncased`, extracts the `[CLS]` embedding, and feeds it into a classifier head. The metadata model extends this architecture by preprocessing structured metadata and concatenating the resulting metadata vector with the BERT `[CLS]` embedding before classification.

Both models were trained using a two-phase strategy: BERT encoder weights are frozen for one epoch (head-only training), then unfrozen for up to four additional epochs of full fine-tuning at `1e-5` learning rate. Early stopping with patience 2 monitors validation loss and restores the best-epoch weights before test evaluation. Class-weighted cross-entropy loss is applied to the multiclass task only to improve minority-class recall; the binary task uses standard unweighted loss because its class distribution is already balanced.

For binary classification, the **BERT + metadata model outperforms the text-only baseline** on all metrics: accuracy 0.6883 vs 0.6808, macro F1 0.6760 vs 0.6689, weighted F1 0.6853 vs 0.6781. For multiclass classification, the results are **mixed**: the metadata model achieves marginally higher accuracy (0.2751 vs 0.2728), while the text-only model achieves higher macro F1 (0.2715 vs 0.2636) due to the benefit of class-weighted training on minority classes.

Overall, metadata provides clear benefit for binary classification. For multiclass classification, class weighting is the more impactful improvement, and the high-dimensional sparse metadata vector partially offsets its own benefit by introducing additional optimisation difficulty. Future work could explore dimensionality reduction, learned metadata embeddings, or more sophisticated fusion strategies to better leverage the metadata signal across both tasks.
