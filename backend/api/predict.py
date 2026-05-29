"""
Prediction API Endpoint
=======================

This module defines the /predict endpoint for the ModelWatch backend.

Each request represents live production traffic and performs:
1. Input validation
2. Model inference
3. Persistent logging of features, outputs, latency, and status
4. JSON response to the client

This endpoint is the primary data source for monitoring and drift detection.
Failed requests are logged with status='failed' rather than crashing the server.

Dependencies:
- FastAPI
- NumPy
- SQLAlchemy
- scikit-learn
- logging
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from sqlalchemy.orm import Session

from backend.services.model_loader import model_service
from backend.database.session import get_db
from backend.database.crud import log_prediction

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    start = time.time()

    try:
        X = np.array(request.features).reshape(1, -1)
        pred, prob = model_service.predict(X)

        latency_ms = int((time.time() - start) * 1000)

        log_prediction(
            db=db,
            model_version=model_service.model_version,
            features=request.features,
            prediction=int(pred[0]),
            fraud_probability=float(prob[0]),
            latency_ms=latency_ms,
            status="success",
        )

        logger.info(
            "prediction logged | version=%s prediction=%d prob=%.4f latency_ms=%d",
            model_service.model_version, int(pred[0]), float(prob[0]), latency_ms
        )

        return PredictionResponse(
            prediction=int(pred[0]),
            fraud_probability=float(prob[0])
        )

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)

        logger.error(
            "prediction failed | version=%s error=%s latency_ms=%d",
            getattr(model_service, "model_version", "unknown"), str(e), latency_ms
        )

        log_prediction(
            db=db,
            model_version=getattr(model_service, "model_version", "unknown"),
            features=request.features,
            prediction=None,
            fraud_probability=None,
            latency_ms=latency_ms,
            status="failed",
            error_message=str(e),
        )

        raise HTTPException(status_code=500, detail="Inference failed. Error has been logged.")
