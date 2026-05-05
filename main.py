from fastapi import FastAPI

app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    openapi_tags=[{"name": "test", "description": "Health check endpoint"}],
)

@app.get("/health", tags=["test"])
def health():
    return {"status": "ok, api is running"}
