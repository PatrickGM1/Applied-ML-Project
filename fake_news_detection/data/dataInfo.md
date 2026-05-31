# LIAR dataset — dataInfo

## Files:

    - train.tsv — 10,269 rows
    - valid.tsv — 1,284 rows
    - test.tsv  — 1,283 rows

## Columns (1–14):

    1. id — JSON filename
    2. label — pants-fire, false, barely-true, half-true, mostly-true, true
    3. statement — raw text
    4. subjects — comma-separated tags
    5. speaker
    6. speaker_job
    7. state
    8. party
    9–13. hist_count_1..hist_count_5 — integers
    14. context

## Label distribution (train):

    - half-true: 2123
    - false: 1998
    - mostly-true: 1966
    - true: 1683
    - barely-true: 1657
    - pants-fire: 842

## Notes:

    - No header: load with sep='\t', header=None and assign column names.
    - Fields 5–8 may be empty.
    - Some rows may contain stray tabs and thus have !=14 fields.

## Processed folders:

    - processed/labeled/ — CSVs after label preparation and integer encoding
    - processed/cleaned_text/ — same CSVs plus the statement_clean column used by the NLP baseline
