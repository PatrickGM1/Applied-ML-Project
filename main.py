import math
import pickle
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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
    statement: str = Field(..., min_length=1, max_length=5000)
    subjects: str = ""
    party: str = "missing"
    state: str = "missing"
    speaker_job: str = "missing"
    hist1: float = 0.0
    hist2: float = 0.0
    hist3: float = 0.0
    hist4: float = 0.0
    hist5: float = 0.0

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
        # Training data stores party in lowercase (e.g. "republican", "democrat").
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

@app.get("/health", tags=["test"])
def health():
    return {
        "status": "ok, api is running",
        "model_loaded": model_bundle is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", tags=["prediction"])
def predict(payload: PredictRequest):
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
        "predicted_label": predicted_label,
        "class_probabilities": class_probabilities,
        "confidence": confidence,
        "is_low_confidence": confidence < 0.60,
        "cleaned_statement": cleaned_statement,
    }
