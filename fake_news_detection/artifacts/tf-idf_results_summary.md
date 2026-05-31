# Model comparison summary (TF‑IDF + Metadata)

## 1) Baseline comparisons (validation)
I compared all of these (text only, metadata only, and text+metadata) and got the following results:

### Multiclass (6 labels)
- `multiclass_text_only`: accuracy=0.2500, f1_macro=0.2313
- `multiclass_metadata_only`: accuracy=0.2866, f1_macro=0.2869
- `multiclass_text_metadata`: accuracy=0.3123, f1_macro=0.3114

### Binary (fake vs real)
- `binary_text_only`: accuracy=0.6708, f1_macro=0.6619
- `binary_metadata_only`: accuracy=0.6408, f1_macro=0.6301
- `binary_text_metadata`: accuracy=0.6809, f1_macro=0.6745

**Conclusion:** best accuracy is for **text+metadata**.

---

## 2) Final evaluation (train+valid → test)
Since text+metadata performed best, in `final_text_metadata_test.py` we trained on **train+valid** and evaluated once on the **held‑out test set**, getting:

- `multiclass_text_metadata_final`: accuracy=0.2744, f1_macro=0.2668
- `binary_text_metadata_final`: accuracy=0.6870, f1_macro=0.6739

---

## 3) What features are used (encoding)
The final model input is a single sparse vector created by concatenating:

1. **Text features (TF‑IDF)**
   - Built from `statement_clean` with word + bigram TF‑IDF.
   - `max_features=20000`, so the vocabulary is capped.

2. **Metadata features**
   - **Subjects**: multi‑hot encoding of comma‑separated `subjects` (each subject tag becomes a 0/1 feature).
   - **Categoricals**: one‑hot encoding for `party`, `state`, `speaker_job` (missing values mapped to a literal `"missing"` category; unseen categories are ignored at inference).
   - **History counts**: numeric features `hist1`…`hist5` (missing treated as 0, then scaled).

---

## 4) How many features in total (and actively used)
“Total features” means the vector length (number of columns). “Active features” means columns that are non‑zero at least once in the **training** matrix.

### Multiclass training split
- train_rows: 10269
- tfidf_features: 15023
- subjects_features: 142
- categorical_features: 1294
- history_features: 5
- **TOTAL features: 16464**
- **ACTIVE features: 16464**

### Binary training split
- train_rows: 6489
- tfidf_features: 9803
- subjects_features: 140
- categorical_features: 1011
- history_features: 5
- **TOTAL features: 10959**
- **ACTIVE features: 10959**

**Note:** Active = Total here, meaning every feature column appears at least once in the training data (no “dead” columns).
