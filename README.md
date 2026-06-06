# Applied Machine Learning project

## Team members:

    - Patrick Gheba
    - Luca Serban
    - Ana-Maria Izbas
    - George Tutui

## Project idea:

    Fake News Detection using the LIAR dataset.
    Dataset: https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset

## Website for deployment:

https://aml.guba.dev/

## Local testing:

### Docker (recommended)

Requires [Docker](https://docs.docker.com/get-docker/) (Desktop on Windows/macOS, Engine on Linux).

```bash
# Windows / macOS
docker compose up --build

# Linux (if your user is not in the docker group)
sudo docker compose up --build
```

| Service            | URL                        |
| ------------------ | -------------------------- |
| FastAPI backend    | http://localhost:8000      |
| Swagger UI         | http://localhost:8000/docs |
| Streamlit frontend | http://localhost:8502      |

Stop with `Ctrl+C`, then `docker compose down`.

---

### Without Docker

**Prerequisites:** Python 3.11+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Start the API
uvicorn main:app --reload --port 8000

# 5. In a second terminal (with the venv activated), start Streamlit
streamlit run streamlit_app.py --server.port 8502
```

The `API_BASE_URL` environment variable controls which backend Streamlit calls (default: `http://localhost:8000`).

---

## API reference

Base URL: `http://localhost:8000` (local) · `https://aml.guba.dev` (production)

Interactive docs (Swagger UI): `GET /docs`

---

### `GET /v1/health`

Check whether the API is reachable and the classification model is loaded.

**Response `200 OK`**

```json
{
  "status": "ok, api is running",
  "model_loaded": true,
  "model_path": "/app/fake_news_detection/artifacts/models/binary_text_metadata_final.joblib"
}
```

---

### `POST /v1/predictions`

Classify a political statement as **real** or **fake**.

The model was trained on the [LIAR dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset) (~12 000 PolitiFact fact-checks). It uses TF-IDF bag-of-words features plus optional speaker metadata.

**Binary labels:**

- `real` - originally rated _true_, _mostly-true_, or _half-true_
- `fake` - originally rated _false_, _barely-true_, or _pants-fire_

All text preprocessing is done server-side so just send the raw statement as-is.

#### Request body (`application/json`)

| Field         | Type        | Required | Default     | Description                                                         |
| ------------- | ----------- | -------- | ----------- | ------------------------------------------------------------------- |
| `statement`   | `string`    | yes      | -           | The political claim to classify (1-5000 chars)                      |
| `subjects`    | `string`    |          | `""`        | Comma-separated topic tags, e.g. `"economy,jobs"`                   |
| `party`       | `string`    |          | `"missing"` | Speaker's party affiliation, e.g. `"republican"`. Case-insensitive. |
| `state`       | `string`    |          | `"missing"` | U.S. state of the speaker, e.g. `"Texas"`                           |
| `speaker_job` | `string`    |          | `"missing"` | Job title at the time of the statement, e.g. `"President"`          |
| `hist1`       | `float ≥ 0` |          | `0.0`       | Speaker's past "barely true" verdict count                          |
| `hist2`       | `float ≥ 0` |          | `0.0`       | Speaker's past "false" verdict count                                |
| `hist3`       | `float ≥ 0` |          | `0.0`       | Speaker's past "half true" verdict count                            |
| `hist4`       | `float ≥ 0` |          | `0.0`       | Speaker's past "mostly true" verdict count                          |
| `hist5`       | `float ≥ 0` |          | `0.0`       | Speaker's past "pants on fire" verdict count                        |

**Minimal example request:**

```json
{
  "statement": "The economy grew by 3% last quarter under my administration."
}
```

**Full example request:**

```json
{
  "statement": "The economy grew by 3% last quarter under my administration.",
  "subjects": "economy,jobs",
  "party": "republican",
  "state": "Texas",
  "speaker_job": "President",
  "hist1": 3,
  "hist2": 7,
  "hist3": 5,
  "hist4": 10,
  "hist5": 1
}
```

#### Response `201 Created`

| Field                 | Type     | Description                                                      |
| --------------------- | -------- | ---------------------------------------------------------------- |
| `label`               | `string` | Predicted class: `"real"` or `"fake"`                            |
| `class_probabilities` | `object` | Probability for each class (values sum to 1.0)                   |
| `confidence`          | `float`  | Probability of the top-ranked class (0.0–1.0)                    |
| `is_low_confidence`   | `bool`   | `true` when `confidence < 0.60`, treat as a weak signal          |
| `cleaned_statement`   | `string` | Statement after server-side preprocessing (useful for debugging) |

```json
{
  "label": "fake",
  "class_probabilities": {
    "fake": 0.73,
    "real": 0.27
  },
  "confidence": 0.73,
  "is_low_confidence": false,
  "cleaned_statement": "economy grow last quarter administration"
}
```

#### Error responses

| Status                      | Cause                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `422 Unprocessable Entity`  | Validation failed: blank statement, negative history count, statement > 5 000 chars, etc. |
| `503 Service Unavailable`   | Model file not found on the server, run the training scripts first                        |
| `500 Internal Server Error` | Unexpected inference error                                                                |

---

### `GET /v2/health`

Check whether the API is reachable and the BERT model is loaded.

**Response `200 OK`**

```json
{
  "status": "ok, api is running",
  "bert_model_loaded": true,
  "model_dir": "/app/fake_news_detection/artifacts/models/bert_text_metadata/binary_bert_metadata"
}
```

---

### `POST /v2/predictions`

Classify a political statement as **real** or **fake** using the BERT + Metadata Fusion model.

This model uses a fine-tuned BERT encoder for text representation, fused with categorical speaker metadata. It does **not** use speaker history counts.

#### Request body (`application/json`)

| Field         | Type     | Required | Default     | Description                                            |
| ------------- | -------- | -------- | ----------- | ------------------------------------------------------ |
| `statement`   | `string` | yes      | -           | The political claim to classify (1-5000 chars)         |
| `subjects`    | `string` |          | `""`        | Comma-separated topic tags, e.g. `"economy,jobs"`      |
| `speaker`     | `string` |          | `"unknown"` | Speaker name, e.g. `"barack-obama"`                    |
| `party`       | `string` |          | `"unknown"` | Speaker's party affiliation, e.g. `"republican"`       |
| `state`       | `string` |          | `"unknown"` | U.S. state of the speaker, e.g. `"Texas"`              |
| `speaker_job` | `string` |          | `"unknown"` | Job title, e.g. `"President"`                          |
| `context`     | `string` |          | `"unknown"` | Context of the statement, e.g. `"a speech"`            |

**Minimal example request:**

```json
{
  "statement": "The economy grew by 3% last quarter under my administration."
}
```

**Full example request:**

```json
{
  "statement": "The economy grew by 3% last quarter under my administration.",
  "subjects": "economy,jobs",
  "speaker": "barack-obama",
  "party": "democrat",
  "state": "Illinois",
  "speaker_job": "President",
  "context": "a speech"
}
```

#### Response `201 Created`

Same schema as v1:

```json
{
  "label": "real",
  "class_probabilities": {
    "fake": 0.33,
    "real": 0.67
  },
  "confidence": 0.67,
  "is_low_confidence": false,
  "cleaned_statement": "The economy grew by 3% last quarter under my administration."
}
```

#### Error responses

| Status                      | Cause                                                        |
| --------------------------- | ------------------------------------------------------------ |
| `422 Unprocessable Entity`  | Validation failed: blank statement, statement > 5000 chars   |
| `503 Service Unavailable`   | BERT model not found, run the training script first          |
| `500 Internal Server Error` | Unexpected inference error                                   |

---

## Models

| Version | Model                        | Features                              | Accuracy |
| ------- | ---------------------------- | ------------------------------------- | -------- |
| v1      | TF-IDF + Logistic Regression | Text + metadata + speaker history     | Baseline |
| v2      | BERT + Metadata Fusion       | BERT text encoder + metadata (no history) | Final    |

---

## Testing

Tests use [pytest](https://docs.pytest.org/) with FastAPI's `TestClient`.

```bash
# Install pytest (if not already)
pip install pytest

# Run all tests
pytest tests/ -v
```

Tests cover: health endpoints, input validation, 503 without models, mocked predictions, response schema, and OpenAPI registration.

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/deploy.yml`):

1. **Test** - runs `pytest tests/ -v` on Python 3.11
2. **Deploy** - only runs if tests pass; SSHs into VPS, pulls latest code, rebuilds Docker containers

---

## Project structure

```
Applied-ML-Project/
├── main.py                          # FastAPI app (v1 + v2 endpoints)
├── streamlit_app.py                 # Streamlit frontend entry point
├── pages/
│   ├── home.py                      # Home page
│   ├── predict.py                   # Base model (v1) demo
│   └── final.py                     # BERT model (v2) demo
├── fake_news_detection/
│   ├── scripts/
│   │   ├── nlp_script.py            # Text preprocessing
│   │   ├── final_text_metadata_test.py  # TF-IDF training script
│   │   ├── bert_text_metadata.py    # BERT training script
│   │   ├── compare_bert_vs_baseline_no_history.py
│   │   └── plot_training_history.py
│   ├── features/
│   │   └── metadata.py              # Metadata feature engineering
│   ├── artifacts/
│   │   ├── models/                  # Trained model weights
│   │   └── encoders/                # Label encoders
│   └── data/
│       └── processed/               # Cleaned CSV data
├── tests/
│   └── test_main.py                 # API tests (pytest)
├── .github/workflows/
│   └── deploy.yml                   # CI/CD pipeline
├── Dockerfile                       # Backend (FastAPI)
├── Dockerfile.streamlit             # Frontend (Streamlit)
├── docker-compose.yml               # Multi-container setup
├── .streamlit/config.toml           # Streamlit theme config
└── requirements.txt
```
