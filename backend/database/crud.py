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
from backend.database.models import PredictionLog, DriftMetric
from datetime import datetime, timezone


def log_prediction(
    db: Session,
    model_version: str,
    features: list,
    prediction: int | None,
    fraud_probability: float | None,
    latency_ms: int,
    status: str,
    error_message: str | None = None,
):
    record = PredictionLog(
        model_version=model_version,
        features=features,
        prediction=prediction,
        fraud_probability=fraud_probability,
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def log_drift_metrics(
    db: Session,
    model_version: str,
    feature_name: str,
    window_size: int,
    ks_statistic: float,
    ks_p_value: float,
    psi_score: float,
    alert: bool,
):
    record = DriftMetric(
        computed_at=datetime.now(timezone.utc),
        model_version=model_version,
        feature_name=feature_name,
        window_size=window_size,
        ks_statistic=ks_statistic,
        ks_p_value=ks_p_value,
        psi_score=psi_score,
        alert=alert,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_predictions(db: Session, limit: int = 500):
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def get_drift_history(db: Session, limit: int = 200):
    return (
        db.query(DriftMetric)
        .order_by(DriftMetric.computed_at.desc())
        .limit(limit)
        .all()
    )