from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake News Detection API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test")
def test():
    return {"status": "ok"}