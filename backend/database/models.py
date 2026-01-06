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

from sqlalchemy import Column, Integer, Float, DateTime, JSON
from datetime import datetime
from backend.database.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Model inputs
    features = Column(JSON)

    # Model outputs
    prediction = Column(Integer)
    fraud_probability = Column(Float)
