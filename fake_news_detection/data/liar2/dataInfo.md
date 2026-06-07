# LIAR2 dataset — dataInfo

## Source

LIAR2 is an enhanced fake news detection benchmark dataset introduced in:

Cheng Xu and M-Tahar Kechadi,
"An Enhanced Fake News Detection System With Fuzzy Deep Learning",
IEEE Access, 2024.

Official repository:
https://github.com/chengxuphd/liar2

## Raw files created by this project

    - data/liar2/raw/train.tsv
    - data/liar2/raw/valid.tsv
    - data/liar2/raw/test.tsv

These files are generated from the official LIAR2 CSV files and converted into a LIAR-like 14-column TSV schema so that the existing project logic remains understandable.

## Processed folders

    - data/processed/liar2/labeled/
      CSVs after label preparation and integer encoding.

    - data/processed/liar2/cleaned_text/
      Same CSVs plus the statement_clean column used by text-based models.

## Expected processed files

    data/processed/liar2/labeled/train.processed.csv
    data/processed/liar2/labeled/valid.processed.csv
    data/processed/liar2/labeled/test.processed.csv

    data/processed/liar2/labeled/train_binary.processed.csv
    data/processed/liar2/labeled/valid_binary.processed.csv
    data/processed/liar2/labeled/test_binary.processed.csv

    data/processed/liar2/cleaned_text/train.processed.csv
    data/processed/liar2/cleaned_text/valid.processed.csv
    data/processed/liar2/cleaned_text/test.processed.csv

    data/processed/liar2/cleaned_text/train_binary.processed.csv
    data/processed/liar2/cleaned_text/valid_binary.processed.csv
    data/processed/liar2/cleaned_text/test_binary.processed.csv

## Label mappings

Multiclass labels:

    0: barely-true
    1: false
    2: half-true
    3: mostly-true
    4: pants-fire
    5: true

Binary labels follow the actual project code in create_labels.py:

    fake:
        - pants-fire
        - false

    real:
        - mostly-true
        - true

The following labels are not used in the binary files:

    - barely-true
    - half-true

This is intentional so that LIAR2 remains comparable with the already trained project binary models.

## Column mapping from LIAR2 to this project

    LIAR2 statement              -> statement
    LIAR2 subject                -> subjects
    LIAR2 speaker                -> speaker
    LIAR2 speaker_description    -> speaker_job
    LIAR2 state_info             -> state
    unavailable party field      -> party = missing
    LIAR2 mostly_false_counts    -> hist1
    LIAR2 false_counts           -> hist2
    LIAR2 half_true_counts       -> hist3
    LIAR2 mostly_true_counts     -> hist4
    LIAR2 pants_on_fire_counts   -> hist5
    LIAR2 context                -> context

## Notes

LIAR2 contains richer information than the original LIAR dataset, including fields such as speaker description and justification. This project-aligned preprocessing keeps only the fields needed by the current project models.
