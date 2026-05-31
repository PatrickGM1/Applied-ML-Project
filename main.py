import math
import pickle
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel, Field, field_validator
from scipy.sparse import hstack

from fake_news_detection.features.metadata import transform_metadata
from fake_news_detection.scripts.nlp_script import clean_text, ensure_nltk_resources

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_detection" / "artifacts" / "models" / "binary_text_metadata_final.joblib"
ENCODER_PATH = BASE_DIR / "fake_news_detection" / "artifacts" / "encoders" / "label_encoder_2.pkl"

model_bundle = None
label_encoder = None

if MODEL_PATH.exists():
    model_bundle = joblib.load(MODEL_PATH)

if ENCODER_PATH.exists():
    with open(ENCODER_PATH, "rb") as file_handle:
        label_encoder = pickle.load(file_handle)

ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="""Classifies political statements as **real** or **fake** using a
TF-IDF + logistic regression model trained on the
[LIAR dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)
(~12 000 PolitiFact fact-checks).

**Binary labels:**
- `real` - originally rated *true*, *mostly-true*, or *half-true*
- `fake` - originally rated *false*, *barely-true*, or *pants-fire*

The model uses TF-IDF bag-of-words features plus optional speaker metadata
(party, state, job title, and how many times the speaker has been rated in each
category before). Text preprocessing is done on the server so just send the
raw statement as-is.""",
    openapi_tags=[
        {"name": "test", "description": "Health check endpoint"},
        {"name": "prediction", "description": "Model inference endpoint"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/demo", include_in_schema=False)
def demo():
    return FileResponse(BASE_DIR / "predict.html")


@app.get("/style.css", include_in_schema=False)
def css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


class PredictRequest(BaseModel):
    """
    Input for the fake-news classifier.

    Only `statement` is required. All other fields are optional and default to
    neutral/missing values, but adding speaker metadata usually helps accuracy.
    """

    statement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw political statement or claim to classify. "
        "Text is preprocessed server-side (lowercased, punctuation removed, "
        "stopwords filtered, lemmatized) so just send it as-is.",
        examples=["The economy grew by 3% last quarter under my administration."],
    )
    subjects: str = Field(
        default="",
        description="Comma-separated topic tags for the statement (e.g. 'economy,jobs'). "
        "Leave blank if unknown.",
        examples=["economy,jobs"],
    )
    party: str = Field(
        default="missing",
        description="Political party affiliation of the speaker "
        "(e.g. 'democrat', 'republican', 'none'). "
        "Gets lowercased automatically.",
        examples=["republican"],
    )
    state: str = Field(
        default="missing",
        description="U.S. state associated with the speaker at the time of the statement "
        "(e.g. 'Texas', 'Illinois'). Use 'missing' if unknown.",
        examples=["Texas"],
    )
    speaker_job: str = Field(
        default="missing",
        description="Job title of the speaker at the time of the statement "
        "(e.g. 'President', 'State senator'). Use 'missing' if unknown.",
        examples=["President"],
    )
    hist1: float = Field(
        default=0.0,
        ge=0,
        description="Speaker's cumulative count of past statements rated 'barely true'.",
        examples=[3],
    )
    hist2: float = Field(
        default=0.0,
        ge=0,
        description="Speaker's cumulative count of past statements rated 'false'.",
        examples=[7],
    )
    hist3: float = Field(
        default=0.0,
        ge=0,
        description="Speaker's cumulative count of past statements rated 'half true'.",
        examples=[5],
    )
    hist4: float = Field(
        default=0.0,
        ge=0,
        description="Speaker's cumulative count of past statements rated 'mostly true'.",
        examples=[10],
    )
    hist5: float = Field(
        default=0.0,
        ge=0,
        description="Speaker's cumulative count of past statements rated 'pants on fire'.",
        examples=[1],
    )

    @field_validator("statement")
    @classmethod
    def statement_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Statement must not be blank or whitespace only.")
        return v

    @field_validator("party")
    @classmethod
    def normalize_party(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("state", "speaker_job", "subjects")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("hist1", "hist2", "hist3", "hist4", "hist5")
    @classmethod
    def history_counts_valid(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("History counts must be finite numbers.")
        if v < 0:
            raise ValueError("History counts must be non-negative.")
        return v


class PredictionResponse(BaseModel):
    """What you get back after classifying a statement."""

    label: str = Field(
        description="Predicted label: either `real` (true/mostly-true/half-true) "
        "or `fake` (false/barely-true/pants-fire).",
        examples=["fake"],
    )
    class_probabilities: dict[str, float] = Field(
        description="Probability for each class. Keys are the label strings, values sum to 1.0.",
        examples=[{"fake": 0.73, "real": 0.27}],
    )
    confidence: float = Field(
        description="Probability of the winning class, same as max(class_probabilities). Between 0 and 1.",
        examples=[0.73],
    )
    is_low_confidence: bool = Field(
        description="True when confidence is below 0.60. Don't trust the result too much in that case.",
        examples=[False],
    )
    cleaned_statement: str = Field(
        description="The statement after server-side preprocessing. Useful for debugging if the prediction looks wrong.",
        examples=["economy grow last quarter administration"],
    )

v1 = APIRouter(prefix="/v1")


@v1.get("/health", tags=["test"])
def health():
    """
    Check if the API is up and the model loaded correctly.

    Returns:
    - **status** - always `"ok, api is running"` if the server is reachable
    - **model_loaded** - `true` if the model file was found and loaded on startup
    - **model_path** - the path it looked for the model at
    """
    return {
        "status": "ok, api is running",
        "model_loaded": model_bundle is not None,
        "model_path": str(MODEL_PATH),
    }


@v1.post("/predictions", tags=["prediction"], status_code=201, response_model=PredictionResponse)
def predict(payload: PredictRequest):
    """
    Classify a political statement as **real** or **fake**.

    Send a statement plus any speaker metadata you have. The server handles all
    the text preprocessing so you don't need to clean it yourself.

    **Required:**
    - `statement` - the claim to classify (1-5000 characters)

    **Optional metadata** (helps accuracy if you have it):
    - `subjects` - comma-separated topic tags (e.g. `"economy,jobs"`)
    - `party` - speaker's party, case-insensitive (e.g. `"republican"`)
    - `state` - U.S. state of the speaker (e.g. `"Texas"`)
    - `speaker_job` - their job title at the time (e.g. `"President"`)
    - `hist1`-`hist5` - how many times the speaker was previously rated:
      barely-true, false, half-true, mostly-true, pants-fire

    **Returns** a `PredictionResponse` with the label, per-class probabilities,
    confidence score, and a flag if confidence is low.

    **Errors:**
    - `422` - bad input (blank statement, negative history count, etc.)
    - `503` - model not loaded on the server
    - `500` - something went wrong during inference
    """
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not found. Train and serialize first by running "
                "fake_news_detection/scripts/final_text_metadata_test.py"
            ),
        )

    cleaned_statement = clean_text(payload.statement, STOP_WORDS, LEMMATIZER)
    if not cleaned_statement:
        raise HTTPException(
            status_code=422,
            detail="Claim text became empty after preprocessing. Please enter a more informative claim.",
        )

    frame = pd.DataFrame(
        [
            {
                "statement_clean": cleaned_statement,
                "subjects": payload.subjects,
                "party": payload.party,
                "state": payload.state,
                "speaker_job": payload.speaker_job,
                "hist1": payload.hist1,
                "hist2": payload.hist2,
                "hist3": payload.hist3,
                "hist4": payload.hist4,
                "hist5": payload.hist5,
            }
        ]
    )

    try:
        x_text = model_bundle["vectorizer"].transform(frame["statement_clean"].fillna(""))
        x_meta = transform_metadata(frame, model_bundle["transformers"])
        x_all = hstack([x_text, x_meta], format="csr")

        predicted_id = int(model_bundle["classifier"].predict(x_all)[0])
        probabilities = model_bundle["classifier"].predict_proba(x_all)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed due to an internal model error: {exc}",
        ) from exc

    if label_encoder is not None:
        predicted_label = str(label_encoder.inverse_transform([predicted_id])[0])
        classes = [str(item) for item in label_encoder.classes_]
    else:
        predicted_label = str(predicted_id)
        classes = [str(item) for item in model_bundle["classifier"].classes_]

    class_probabilities = {
        classes[index]: float(probabilities[index]) for index in range(len(classes))
    }
    confidence = max(class_probabilities.values()) if class_probabilities else 0.0

    return {
        "label": predicted_label,
        "class_probabilities": class_probabilities,
        "confidence": confidence,
        "is_low_confidence": confidence < 0.60,
        "cleaned_statement": cleaned_statement,
    }


app.include_router(v1)
