"""
Prediction Database Models
==========================

This module defines the database schema used by the ModelWatch backend.

Each row in the Prediction table represents a single model inference
performed in production. These records form the historical dataset
used for monitoring data drift and model performance over time.

Dependencies:
- SQLAlchemy
- datetime
"""

from sqlalchemy import Column, Integer, Float, DateTime, String, Boolean, JSON
from datetime import datetime, timezone
from backend.database.session import Base


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    model_version = Column(String(64), nullable=False, index=True)
    features = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=True)       # NULL if inference failed
    fraud_probability = Column(Float, nullable=True)  # NULL if inference failed
    latency_ms = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False)        # success | failed | validation_error
    error_message = Column(String, nullable=True)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    model_version = Column(String(64), nullable=False)
    feature_name = Column(String(128), nullable=False)
    window_size = Column(Integer, nullable=False)
    ks_statistic = Column(Float, nullable=False)
    ks_p_value = Column(Float, nullable=False)
    psi_score = Column(Float, nullable=False)
    alert = Column(Boolean, nullable=False)