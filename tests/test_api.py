"""
API Tests
=========
Basic smoke tests for the ModelWatch FastAPI endpoints.

Tests:
- Health check returns 200
- Predict endpoint returns 200 with valid input
- Predict endpoint returns 422 with invalid input
- Drift endpoint returns 200 or 400 (baselines may not exist in CI)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_input():
    features = [0.1] * 29
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_invalid_input():
    response = client.post("/predict", json={"features": "not_a_list"})
    assert response.status_code == 422


def test_predict_empty_features():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_drift_endpoint_returns():
    response = client.get("/drift/")
    # Either 200 (computed) or 400 (baselines missing in CI) are valid
    assert response.status_code in [200, 400]