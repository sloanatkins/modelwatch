"""
Prediction API Endpoint
=======================

This module defines the /predict endpoint for the ModelWatch backend.

Each request represents live production traffic and performs:
1. Input validation
2. Model inference
3. Persistent logging of features and outputs
4. JSON response to the client

This endpoint is the primary data source for monitoring and drift detection.

Dependencies:
- FastAPI
- NumPy
- SQLAlchemy
- scikit-learn
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
import numpy as np
from sqlalchemy.orm import Session

from backend.services.model_loader import model_service
from backend.database.session import SessionLocal
from backend.database.crud import log_prediction

router = APIRouter()


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    X = np.array(request.features).reshape(1, -1)

    pred, prob = model_service.predict(X)

    # Persist full inference record
    log_prediction(
        db=db,
        features=request.features,
        prediction=int(pred[0]),
        fraud_probability=float(prob[0])
    )

    return PredictionResponse(
        prediction=int(pred[0]),
        fraud_probability=float(prob[0])
    )
