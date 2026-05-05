from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake News Detection API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float | None = None
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}