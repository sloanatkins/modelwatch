"""
Database CRUD Operations
========================

This module contains helper functions for interacting with the database.

The API layer should never write raw SQL. All database interactions
are abstracted through this file to ensure consistency and safety.

Dependencies:
- SQLAlchemy ORM
"""

from sqlalchemy.orm import Session
from backend.database.models import Prediction


def log_prediction(
    db: Session,
    features: list,
    prediction: int,
    fraud_probability: float
):
    record = Prediction(
        features=features,
        prediction=prediction,
        fraud_probability=fraud_probability
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_recent_predictions(db: Session, limit: int = 500):
    """
    Returns the most recent prediction rows (used for drift checks).
    """
    return (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
