import io
import re
import urllib.request
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Original LIAR raw folder, already present in your project.
# Used only to safely infer numeric LIAR2 label meanings if needed.
ORIGINAL_LIAR_RAW_DIR = PROJECT_DIR / "data" / "raw"

# New LIAR2-specific raw folder.
LIAR2_BASE_DIR = PROJECT_DIR / "data" / "liar2"
LIAR2_RAW_DIR = LIAR2_BASE_DIR / "raw"

# New LIAR2-specific processed folders.
LIAR2_LABELED_DIR = PROJECT_DIR / "data" / "liar2"/ "processed" / "labeled"
LIAR2_CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "liar2"/ "processed" / "cleaned_text"

# LIAR2_LABELED_DIR = PROJECT_DIR / "data" / "processed" / "liar2" / "labeled"
# LIAR2_CLEANED_TEXT_DIR = PROJECT_DIR / "data" / "processed" / "liar2" / "cleaned_text"

LIAR2_SOURCE_URLS = {
    "train": "https://raw.githubusercontent.com/chengxuphd/liar2/main/liar2/train.csv",
    "valid": "https://raw.githubusercontent.com/chengxuphd/liar2/main/liar2/valid.csv",
    "test": "https://raw.githubusercontent.com/chengxuphd/liar2/main/liar2/test.csv",
}

RAW_LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "hist1",
    "hist2",
    "hist3",
    "hist4",
    "hist5",
    "context",
]

PROJECT_LABEL6_ORDER = [
    "barely-true",
    "false",
    "half-true",
    "mostly-true",
    "pants-fire",
    "true",
]

PROJECT_LABEL6_TO_INT = {
    label_name: index for index, label_name in enumerate(PROJECT_LABEL6_ORDER)
}

# Important: this follows your actual create_labels.py logic.
# It does NOT use the broader README binary mapping.
PROJECT_BINARY_LABEL_MAP = {
    "pants-fire": "fake",
    "false": "fake",
    "mostly-true": "real",
    "true": "real",
}

PROJECT_BINARY_TO_INT = {
    "fake": 0,
    "real": 1,
}

NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def make_directories():
    LIAR2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    LIAR2_LABELED_DIR.mkdir(parents=True, exist_ok=True)
    LIAR2_CLEANED_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def download_csv(split_name: str) -> pd.DataFrame:
    url = LIAR2_SOURCE_URLS[split_name]
    print(f"Downloading LIAR2 {split_name}.csv from:")
    print(f"  {url}")

    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()

    frame = pd.read_csv(io.BytesIO(content))
    frame.columns = [str(column).strip() for column in frame.columns]

    return frame


def normalize_statement_for_matching(text) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label_text(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    replacements = {
        "pants on fire": "pants-fire",
        "pants-on-fire": "pants-fire",
        "pants fire": "pants-fire",
        "barely true": "barely-true",
        "barely_true": "barely-true",
        "half true": "half-true",
        "half_true": "half-true",
        "mostly true": "mostly-true",
        "mostly_true": "mostly-true",
        "true": "true",
        "false": "false",
    }

    return replacements.get(value, value)


def load_original_liar_for_label_inference() -> pd.DataFrame:
    frames = []

    for file_name in ["train.tsv", "valid.tsv", "test.tsv"]:
        file_path = ORIGINAL_LIAR_RAW_DIR / file_name

        if not file_path.exists():
            continue

        frame = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=RAW_LIAR_COLUMNS,
            engine="python",
            quoting=3,
            dtype=str,
        )

        frames.append(frame[["statement", "label"]])

    if not frames:
        raise FileNotFoundError(
            "Could not find the original LIAR raw train.tsv, valid.tsv, or test.tsv files.\n"
            "They are needed only if LIAR2 labels are numeric and must be inferred safely."
        )

    original = pd.concat(frames, ignore_index=True)
    original["statement_key"] = original["statement"].map(normalize_statement_for_matching)
    original = original.drop_duplicates(subset=["statement_key"])

    return original


def infer_numeric_label_map(all_liar2_frames: dict[str, pd.DataFrame]) -> dict[int, str]:
    """
    LIAR2 source files may store labels as integers.
    Instead of guessing what 0, 1, 2, ... mean, we infer the mapping by matching
    overlapping statements against the original LIAR dataset already in this project.
    """
    original = load_original_liar_for_label_inference()

    liar2_combined = []

    for split_name, frame in all_liar2_frames.items():
        temp = frame[["statement", "label"]].copy()
        temp["split"] = split_name
        temp["statement_key"] = temp["statement"].map(normalize_statement_for_matching)
        liar2_combined.append(temp)

    liar2_combined = pd.concat(liar2_combined, ignore_index=True)

    merged = liar2_combined.merge(
        original[["statement_key", "label"]],
        on="statement_key",
        how="inner",
        suffixes=("_liar2", "_original"),
    )

    if merged.empty:
        raise ValueError(
            "LIAR2 labels look numeric, but the script could not infer their meaning "
            "because no overlapping statements were found with the original LIAR raw files."
        )

    label_map = {}

    print("\nInferred LIAR2 numeric label mapping:")

    for numeric_label, group in merged.groupby("label_liar2"):
        counts = Counter(group["label_original"])
        best_label, best_count = counts.most_common(1)[0]

        numeric_label = int(numeric_label)
        label_map[numeric_label] = best_label

        agreement = best_count / len(group)

        print(
            f"  {numeric_label} -> {best_label} "
            f"based on {best_count}/{len(group)} overlapping rows "
            f"(agreement={agreement:.3f})"
        )

    print()

    return label_map


def get_liar2_label_names(
    frame: pd.DataFrame,
    numeric_label_map: dict[int, str] | None,
) -> pd.Series:
    raw_labels = frame["label"]

    normalized_text_labels = raw_labels.map(normalize_label_text)

    known_labels = set(PROJECT_LABEL6_ORDER)

    if set(normalized_text_labels.dropna().unique()).issubset(known_labels):
        return normalized_text_labels

    numeric_labels = pd.to_numeric(raw_labels, errors="raise").astype(int)

    if numeric_label_map is None:
        raise ValueError(
            "LIAR2 labels appear to be numeric, but numeric_label_map was not provided."
        )

    label_names = numeric_labels.map(numeric_label_map)

    if label_names.isna().any():
        unknown = sorted(numeric_labels[label_names.isna()].unique().tolist())
        raise ValueError(f"Could not map these LIAR2 numeric labels: {unknown}")

    return label_names


def get_column(frame: pd.DataFrame, possible_names: list[str], default_value):
    for column_name in possible_names:
        if column_name in frame.columns:
            return frame[column_name]

    return pd.Series([default_value] * len(frame), index=frame.index)


def convert_liar2_to_project_raw_schema(
    source_frame: pd.DataFrame,
    split_name: str,
    numeric_label_map: dict[int, str] | None,
) -> pd.DataFrame:
    """
    Converts official LIAR2 columns to a LIAR-like 14-column schema.

    This produces files similar in shape to the original LIAR raw TSV files:
      id, label, statement, subjects, speaker, speaker_job, state, party,
      hist1, hist2, hist3, hist4, hist5, context
    """
    frame = pd.DataFrame(index=source_frame.index)

    frame["id"] = get_column(
        source_frame,
        ["id", "ID", "json_id"],
        None,
    )

    if frame["id"].isna().all():
        frame["id"] = [f"liar2_{split_name}_{index}" for index in range(len(source_frame))]
    else:
        frame["id"] = frame["id"].fillna("").astype(str)

    frame["label"] = get_liar2_label_names(source_frame, numeric_label_map)

    frame["statement"] = get_column(
        source_frame,
        ["statement", "Statement"],
        "",
    ).fillna("").astype(str)

    frame["subjects"] = (
        get_column(source_frame, ["subjects", "subject", "Subject"], "")
        .fillna("")
        .astype(str)
        .str.replace(";", ",", regex=False)
    )

    frame["speaker"] = get_column(
        source_frame,
        ["speaker", "Speaker"],
        "missing",
    ).fillna("missing").astype(str)

    frame["speaker_job"] = get_column(
        source_frame,
        ["speaker_job", "speaker_description", "Speaker Description"],
        "missing",
    ).fillna("missing").astype(str)

    frame["state"] = get_column(
        source_frame,
        ["state", "state_info", "State Info"],
        "missing",
    ).fillna("missing").astype(str)

    # LIAR2 does not provide party affiliation in the same way as original LIAR.
    frame["party"] = "missing"

    # Original LIAR:
    # hist1 = barely-true count
    # hist2 = false count
    # hist3 = half-true count
    # hist4 = mostly-true count
    # hist5 = pants-fire count
    #
    # LIAR2 commonly uses mostly_false_counts instead of barely_true_counts.
    frame["hist1"] = pd.to_numeric(
        get_column(source_frame, ["barely_true_counts", "mostly_false_counts"], 0),
        errors="coerce",
    ).fillna(0).astype(int)

    frame["hist2"] = pd.to_numeric(
        get_column(source_frame, ["false_counts"], 0),
        errors="coerce",
    ).fillna(0).astype(int)

    frame["hist3"] = pd.to_numeric(
        get_column(source_frame, ["half_true_counts"], 0),
        errors="coerce",
    ).fillna(0).astype(int)

    frame["hist4"] = pd.to_numeric(
        get_column(source_frame, ["mostly_true_counts"], 0),
        errors="coerce",
    ).fillna(0).astype(int)

    frame["hist5"] = pd.to_numeric(
        get_column(source_frame, ["pants_on_fire_counts", "pants_fire_counts"], 0),
        errors="coerce",
    ).fillna(0).astype(int)

    frame["context"] = get_column(
        source_frame,
        ["context", "Context"],
        "",
    ).fillna("").astype(str)

    return frame[RAW_LIAR_COLUMNS]


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()

    labeled["label_binary"] = labeled["label"].map(PROJECT_BINARY_LABEL_MAP)
    labeled["label6_int"] = labeled["label"].map(PROJECT_LABEL6_TO_INT)

    return labeled


def build_binary_subset(frame: pd.DataFrame) -> pd.DataFrame:
    binary = frame[frame["label_binary"].notna()].copy()
    binary["label2_int"] = binary["label_binary"].map(PROJECT_BINARY_TO_INT)

    # Keep binary files focused on the binary task.
    # label6_int is not needed there, but keeping it is harmless.
    return binary


def ensure_nltk_resources():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def clean_text(text, stop_words, lemmatizer) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    cleaned_tokens = []

    for token in text.split():
        if token in stop_words:
            continue

        cleaned_tokens.append(lemmatizer.lemmatize(token))

    return " ".join(cleaned_tokens)


def save_raw_tsv(split_name: str, frame: pd.DataFrame):
    path = LIAR2_RAW_DIR / f"{split_name}.tsv"

    frame.to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
    )

    print(f"Saved raw LIAR2 TSV: {path}")


def save_labeled_processed(split_name: str, multiclass_frame: pd.DataFrame, binary_frame: pd.DataFrame):
    multiclass_path = LIAR2_LABELED_DIR / f"{split_name}.processed.csv"
    binary_path = LIAR2_LABELED_DIR / f"{split_name}_binary.processed.csv"

    multiclass_frame.to_csv(multiclass_path, index=False)
    binary_frame.to_csv(binary_path, index=False)

    print(f"Saved labeled multiclass file: {multiclass_path}")
    print(f"Saved labeled binary file    : {binary_path}")


def save_cleaned_processed(split_name: str, multiclass_frame: pd.DataFrame, binary_frame: pd.DataFrame):
    multiclass_path = LIAR2_CLEANED_TEXT_DIR / f"{split_name}.processed.csv"
    binary_path = LIAR2_CLEANED_TEXT_DIR / f"{split_name}_binary.processed.csv"

    multiclass_frame.to_csv(multiclass_path, index=False)
    binary_frame.to_csv(binary_path, index=False)

    print(f"Saved cleaned multiclass file: {multiclass_path}")
    print(f"Saved cleaned binary file    : {binary_path}")


def write_readme_files():
    raw_readme = """LIAR2: ENHANCED FAKE NEWS DETECTION DATASET

Source:
Cheng Xu and M-Tahar Kechadi, "An Enhanced Fake News Detection System With Fuzzy Deep Learning",
IEEE Access, 2024.

The files in this folder were generated by:
    fake_news_detection/scripts/prepare_liar2_data.py

They are derived from the official LIAR2 CSV files:
    https://github.com/chengxuphd/liar2/tree/main/liar2

=====================================================================
Description of the TSV format used in this project:

Column 1: the generated or provided ID of the statement.
Column 2: the six-class truthfulness label.
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker description, mapped to the original project's speaker_job column.
Column 7: the state information.
Column 8: the party affiliation. LIAR2 does not provide this in the same format, so this is set to "missing".
Column 9-13: the speaker's credibility history counts, aligned to the original LIAR-style metadata fields.
9: barely-true / mostly-false count.
10: false count.
11: half-true count.
12: mostly-true count.
13: pants-on-fire count.
Column 14: the context.

=====================================================================
Important notes:

- These TSV files are project-aligned LIAR2 raw files.
- They are not meant to overwrite the original LIAR raw files under data/raw/.
- The original project's binary mapping is preserved:
    fake = pants-fire + false
    real = mostly-true + true
- barely-true and half-true are excluded from the binary processed files, because the existing project binary models were trained that way.
"""

    data_info = """# LIAR2 dataset — dataInfo

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
"""

    raw_readme_path = LIAR2_RAW_DIR / "README"
    data_info_path = LIAR2_BASE_DIR / "dataInfo.md"

    raw_readme_path.write_text(raw_readme, encoding="utf-8")
    data_info_path.write_text(data_info, encoding="utf-8")

    print(f"Saved README: {raw_readme_path}")
    print(f"Saved dataInfo: {data_info_path}")


def print_summary(split_name: str, multiclass_frame: pd.DataFrame, binary_frame: pd.DataFrame):
    print()
    print(f"Summary for LIAR2 {split_name}:")
    print(f"  multiclass rows: {len(multiclass_frame)}")
    print(f"  binary rows    : {len(binary_frame)}")
    print("  multiclass label distribution:")
    print(multiclass_frame["label"].value_counts().sort_index())
    print("  binary label distribution:")
    print(binary_frame["label_binary"].value_counts().sort_index())
    print()


def main():
    make_directories()

    source_frames = {
        split_name: download_csv(split_name)
        for split_name in ["train", "valid", "test"]
    }

    # If labels are numeric, infer the mapping once using all splits.
    # If labels are already textual, this will remain None and will not be used.
    numeric_label_map = None

    try:
        sample_labels = pd.concat(
            [frame["label"] for frame in source_frames.values()],
            ignore_index=True,
        )
        normalized_sample = sample_labels.map(normalize_label_text)
        known_labels = set(PROJECT_LABEL6_ORDER)

        if not set(normalized_sample.dropna().unique()).issubset(known_labels):
            numeric_label_map = infer_numeric_label_map(source_frames)

    except Exception as error:
        raise RuntimeError(
            "Could not determine LIAR2 label format safely."
        ) from error

    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    for split_name in ["train", "valid", "test"]:
        raw_project_frame = convert_liar2_to_project_raw_schema(
            source_frames[split_name],
            split_name=split_name,
            numeric_label_map=numeric_label_map,
        )

        save_raw_tsv(split_name, raw_project_frame)

        multiclass_frame = add_labels(raw_project_frame)
        binary_frame = build_binary_subset(multiclass_frame)

        save_labeled_processed(split_name, multiclass_frame, binary_frame)

        multiclass_cleaned = multiclass_frame.copy()
        binary_cleaned = binary_frame.copy()

        multiclass_cleaned["statement_clean"] = multiclass_cleaned["statement"].map(
            lambda text: clean_text(text, stop_words, lemmatizer)
        )

        binary_cleaned["statement_clean"] = binary_cleaned["statement"].map(
            lambda text: clean_text(text, stop_words, lemmatizer)
        )

        save_cleaned_processed(split_name, multiclass_cleaned, binary_cleaned)
        print_summary(split_name, multiclass_frame, binary_frame)

    write_readme_files()

    print()
    print("LIAR2 data preparation complete.")
    print()
    print("Raw LIAR2 TSV files:")
    print(f"  {LIAR2_RAW_DIR}")
    print()
    print("Processed LIAR2 labeled files:")
    print(f"  {LIAR2_LABELED_DIR}")
    print()
    print("Processed LIAR2 cleaned text files:")
    print(f"  {LIAR2_CLEANED_TEXT_DIR}")


if __name__ == "__main__":
    main()