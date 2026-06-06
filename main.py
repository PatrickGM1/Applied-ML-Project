import json
import math
import pickle
from pathlib import Path

import joblib
import numpy as np
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

BERT_MODEL_DIR = BASE_DIR / "fake_news_detection" / "artifacts" / "models" / "bert_text_metadata" / "binary_bert_metadata"

model_bundle = None
label_encoder = None
bert_bundle = None

if MODEL_PATH.exists():
    model_bundle = joblib.load(MODEL_PATH)

if ENCODER_PATH.exists():
    with open(ENCODER_PATH, "rb") as file_handle:
        label_encoder = pickle.load(file_handle)

# --- Load BERT model if available ---
if (BERT_MODEL_DIR / "serving_config.json").exists():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    with open(BERT_MODEL_DIR / "serving_config.json", encoding="utf-8") as _fh:
        _bert_cfg = json.load(_fh)

    class _BertMetadataFusion(nn.Module):
        def __init__(self, bert_model_name, meta_dim, num_labels, hidden_dim, dropout_rate):
            super().__init__()
            self.bert = AutoModel.from_pretrained(bert_model_name)
            bert_dim = self.bert.config.hidden_size
            self.fusion_head = nn.Sequential(
                nn.Linear(bert_dim + meta_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_labels),
            )

        def forward(self, input_ids, attention_mask, token_type_ids, metadata):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            fused = torch.cat([cls_embedding, metadata], dim=-1)
            return self.fusion_head(fused)

    _bert_model = _BertMetadataFusion(
        bert_model_name=_bert_cfg["model_name"],
        meta_dim=_bert_cfg["meta_dim"],
        num_labels=_bert_cfg["num_labels"],
        hidden_dim=_bert_cfg["hidden_dim"],
        dropout_rate=_bert_cfg["dropout_rate"],
    )
    _bert_model.load_state_dict(
        torch.load(BERT_MODEL_DIR / "model_weights.pt", map_location="cpu", weights_only=True)
    )
    _bert_model.eval()

    _bert_tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))

    with open(BERT_MODEL_DIR / "meta_transformers.pkl", "rb") as _fh:
        _bert_meta_transformers = pickle.load(_fh)

    _bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bert_model.to(_bert_device)

    bert_bundle = {
        "model": _bert_model,
        "tokenizer": _bert_tokenizer,
        "meta_transformers": _bert_meta_transformers,
        "config": _bert_cfg,
        "device": _bert_device,
    }
    print(f"BERT model loaded from {BERT_MODEL_DIR} (device: {_bert_device})")

ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

app = FastAPI(
    title="Fake News Detection API",
    version="2.0.0",
    description="""Classifies political statements as **real** or **fake** using models
trained on the [LIAR dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)
(~12 000 PolitiFact fact-checks).

**Binary labels:**
- `real` - originally rated *true*, *mostly-true*, or *half-true*
- `fake` - originally rated *false*, *barely-true*, or *pants-fire*

**Two model versions:**
- **v1** — TF-IDF + Logistic Regression baseline (text + metadata + speaker history)
- **v2** — BERT + Metadata Fusion (text + metadata, no speaker history)""",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "v1 - baseline", "description": "TF-IDF + Logistic Regression (text + metadata + history)"},
        {"name": "v2 - bert", "description": "BERT + Metadata Fusion (text + metadata, no history)"},
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


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """What you get back after classifying a statement."""

    label: str = Field(
        description="Predicted label: either `real` or `fake`.",
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
        description="True when confidence is below 0.60.",
        examples=[False],
    )
    cleaned_statement: str = Field(
        description="The statement after preprocessing.",
        examples=["economy grow last quarter administration"],
    )


# ===================================================================
#  V1 — TF-IDF + Logistic Regression baseline
# ===================================================================

class V1PredictRequest(BaseModel):
    """
    Input for the TF-IDF baseline classifier.

    Only `statement` is required. All other fields are optional and default to
    neutral/missing values, but adding speaker metadata usually helps accuracy.
    """

    statement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw political statement or claim to classify.",
        examples=["The economy grew by 3% last quarter under my administration."],
    )
    subjects: str = Field(
        default="",
        description="Comma-separated topic tags (e.g. 'economy,jobs').",
        examples=["economy,jobs"],
    )
    party: str = Field(
        default="missing",
        description="Political party of the speaker.",
        examples=["republican"],
    )
    state: str = Field(
        default="missing",
        description="U.S. state of the speaker.",
        examples=["Texas"],
    )
    speaker_job: str = Field(
        default="missing",
        description="Job title of the speaker.",
        examples=["President"],
    )
    hist1: float = Field(default=0.0, ge=0, description="Past 'barely true' count.", examples=[3])
    hist2: float = Field(default=0.0, ge=0, description="Past 'false' count.", examples=[7])
    hist3: float = Field(default=0.0, ge=0, description="Past 'half true' count.", examples=[5])
    hist4: float = Field(default=0.0, ge=0, description="Past 'mostly true' count.", examples=[10])
    hist5: float = Field(default=0.0, ge=0, description="Past 'pants on fire' count.", examples=[1])

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


v1 = APIRouter(prefix="/v1")


@v1.get("/health", tags=["health"])
def health_v1():
    """Check if the API and v1 baseline model are loaded."""
    return {
        "status": "ok, api is running",
        "model_loaded": model_bundle is not None,
        "model_path": str(MODEL_PATH),
    }


@v1.post("/predictions", tags=["v1 - baseline"], status_code=201, response_model=PredictionResponse)
def predict_v1(payload: V1PredictRequest):
    """
    Classify a statement using the TF-IDF + Logistic Regression baseline.

    Uses TF-IDF text features + speaker metadata including history counts.
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


# ===================================================================
#  V2 — BERT + Metadata Fusion
# ===================================================================

class V2PredictRequest(BaseModel):
    """
    Input for the BERT + metadata fusion model.

    Only `statement` is required. Speaker metadata helps accuracy.
    No history counts — this model does not use speaker history.
    """

    statement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw political statement or claim to classify.",
        examples=["The economy grew by 3% last quarter under my administration."],
    )
    subjects: str = Field(default="", description="Comma-separated topic tags.", examples=["economy,jobs"])
    speaker: str = Field(default="unknown", description="Speaker name.", examples=["barack-obama"])
    party: str = Field(default="unknown", description="Political party.", examples=["republican"])
    state: str = Field(default="unknown", description="U.S. state.", examples=["Texas"])
    speaker_job: str = Field(default="unknown", description="Job title.", examples=["President"])
    context: str = Field(default="unknown", description="Context of the statement.", examples=["a speech"])

    @field_validator("statement")
    @classmethod
    def statement_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Statement must not be blank or whitespace only.")
        return v

    @field_validator("party", "state", "speaker_job", "subjects", "speaker", "context")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


v2 = APIRouter(prefix="/v2")


@v2.get("/health", tags=["health"])
def health_v2():
    """Check if the API and v2 BERT model are loaded."""
    return {
        "status": "ok, api is running",
        "bert_model_loaded": bert_bundle is not None,
        "model_dir": str(BERT_MODEL_DIR),
    }


@v2.post("/predictions", tags=["v2 - bert"], status_code=201, response_model=PredictionResponse)
def predict_v2(payload: V2PredictRequest):
    """
    Classify a statement using the BERT + metadata fusion model.

    Uses BERT text embeddings fused with speaker metadata (no history features).
    """
    if bert_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "BERT model not found. Train and serialize first by running "
                "fake_news_detection/scripts/bert_text_metadata.py"
            ),
        )

    import torch

    model = bert_bundle["model"]
    tokenizer = bert_bundle["tokenizer"]
    meta_transformers = bert_bundle["meta_transformers"]
    cfg = bert_bundle["config"]
    device = bert_bundle["device"]

    frame = pd.DataFrame([{
        "statement": payload.statement.strip(),
        "subjects": payload.subjects,
        "speaker": payload.speaker,
        "party": payload.party,
        "state": payload.state,
        "speaker_job": payload.speaker_job,
        "context": payload.context,
    }])

    try:
        metadata = meta_transformers.transform(frame)
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32).to(device)

        enc = tokenizer(
            payload.statement.strip(),
            truncation=True,
            padding="max_length",
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get(
            "token_type_ids",
            torch.zeros_like(enc["input_ids"]),
        ).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids, metadata_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predicted_id = int(np.argmax(probs))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BERT prediction failed: {exc}",
        ) from exc

    if label_encoder is not None:
        predicted_label = str(label_encoder.inverse_transform([predicted_id])[0])
        classes = [str(c) for c in label_encoder.classes_]
    else:
        classes = [str(i) for i in range(cfg["num_labels"])]
        predicted_label = classes[predicted_id]

    class_probabilities = {
        classes[i]: float(probs[i]) for i in range(len(classes))
    }
    confidence = max(class_probabilities.values()) if class_probabilities else 0.0

    return {
        "label": predicted_label,
        "class_probabilities": class_probabilities,
        "confidence": confidence,
        "is_low_confidence": confidence < 0.60,
        "cleaned_statement": payload.statement.strip(),
    }


app.include_router(v1)
app.include_router(v2)
