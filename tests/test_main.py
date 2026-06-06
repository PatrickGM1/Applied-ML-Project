"""API tests for v1 (TF-IDF baseline) and v2 (BERT) endpoints.

These tests use FastAPI's TestClient and do NOT require model files.
When models are absent the API returns 503 — we test that too.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Fresh app import per test to avoid module-level model loading issues."""
    import importlib
    import main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


# ── Health endpoints ──

def test_v1_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok, api is running"
    assert "model_loaded" in body


def test_v2_health(client):
    resp = client.get("/v2/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok, api is running"
    assert "bert_model_loaded" in body


# ── Schema validation ──

def test_v1_empty_statement_rejected(client):
    resp = client.post("/v1/predictions", json={"statement": ""})
    assert resp.status_code == 422


def test_v1_missing_statement_rejected(client):
    resp = client.post("/v1/predictions", json={})
    assert resp.status_code == 422


def test_v2_empty_statement_rejected(client):
    resp = client.post("/v2/predictions", json={"statement": ""})
    assert resp.status_code == 422


def test_v2_missing_statement_rejected(client):
    resp = client.post("/v2/predictions", json={})
    assert resp.status_code == 422


def test_v1_negative_history_rejected(client):
    resp = client.post("/v1/predictions", json={
        "statement": "test claim",
        "hist1": -5,
    })
    assert resp.status_code == 422


def test_v1_whitespace_statement_rejected(client):
    resp = client.post("/v1/predictions", json={"statement": "   "})
    assert resp.status_code == 422


def test_v2_whitespace_statement_rejected(client):
    resp = client.post("/v2/predictions", json={"statement": "   "})
    assert resp.status_code == 422


# ── 503 when models not loaded ──

def test_v1_503_without_model(client):
    import main as main_mod
    main_mod.model_bundle = None
    resp = client.post("/v1/predictions", json={
        "statement": "The economy grew by 3 percent last quarter",
    })
    assert resp.status_code == 503
    assert "not found" in resp.json()["detail"].lower()


def test_v2_503_without_model(client):
    import main as main_mod
    main_mod.bert_bundle = None
    resp = client.post("/v2/predictions", json={
        "statement": "The economy grew by 3 percent last quarter",
    })
    assert resp.status_code == 503
    assert "not found" in resp.json()["detail"].lower()


# ── V1 prediction with mocked model ──

def test_v1_prediction_mocked(client):
    import main as main_mod

    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([1])
    mock_classifier.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_classifier.classes_ = np.array([0, 1])

    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    mock_transformers = MagicMock()

    main_mod.model_bundle = {
        "classifier": mock_classifier,
        "vectorizer": mock_vectorizer,
        "transformers": mock_transformers,
    }

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = np.array(["real"])
    mock_encoder.classes_ = np.array(["fake", "real"])
    main_mod.label_encoder = mock_encoder

    resp = client.post("/v1/predictions", json={
        "statement": "The economy grew by 3 percent last quarter",
        "party": "democrat",
        "state": "ohio",
    })

    assert resp.status_code == 201
    body = resp.json()
    assert body["label"] in ("real", "fake")
    assert "class_probabilities" in body
    assert "confidence" in body
    assert "is_low_confidence" in body
    assert "cleaned_statement" in body
    assert 0 <= body["confidence"] <= 1


# ── V1 defaults are applied ──

def test_v1_defaults(client):
    """Only statement required — everything else has defaults."""
    import main as main_mod

    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([0])
    mock_classifier.predict_proba.return_value = np.array([[0.6, 0.4]])
    mock_classifier.classes_ = np.array([0, 1])

    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    main_mod.model_bundle = {
        "classifier": mock_classifier,
        "vectorizer": mock_vectorizer,
        "transformers": MagicMock(),
    }

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = np.array(["fake"])
    mock_encoder.classes_ = np.array(["fake", "real"])
    main_mod.label_encoder = mock_encoder

    resp = client.post("/v1/predictions", json={
        "statement": "taxes will double next year",
    })
    assert resp.status_code == 201


# ── Response schema shape ──

def test_response_schema_keys(client):
    import main as main_mod

    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([0])
    mock_classifier.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_classifier.classes_ = np.array([0, 1])

    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    main_mod.model_bundle = {
        "classifier": mock_classifier,
        "vectorizer": mock_vectorizer,
        "transformers": MagicMock(),
    }

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = np.array(["fake"])
    mock_encoder.classes_ = np.array(["fake", "real"])
    main_mod.label_encoder = mock_encoder

    resp = client.post("/v1/predictions", json={
        "statement": "testing schema",
    })

    assert resp.status_code == 201
    body = resp.json()
    expected_keys = {"label", "class_probabilities", "confidence", "is_low_confidence", "cleaned_statement"}
    assert set(body.keys()) == expected_keys


# ── OpenAPI docs available ──

def test_openapi_docs(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_json(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema["paths"]
    assert "/v1/predictions" in paths
    assert "/v2/predictions" in paths
    assert "/v1/health" in paths
    assert "/v2/health" in paths
