"""
Feature Window Extraction
=========================

This module provides utilities for extracting recent feature windows
from the prediction database.

Rolling windows are the foundation of statistical drift detection,
allowing comparisons between training baselines and live data.

Dependencies:
- SQLAlchemy
- NumPy
"""

import numpy as np
from sqlalchemy.orm import Session
from backend.database.models import Prediction


def get_recent_feature_window(
    db: Session,
    window_size: int = 500
) -> np.ndarray:
    """
    Retrieve the most recent feature vectors from the database.

    Parameters:
    -----------
    db : Session
        Active database session
    window_size : int
        Number of recent predictions to retrieve

    Returns:
    --------
    np.ndarray
        Array of shape (window_size, n_features)
    """

    records = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(window_size)
        .all()
    )

    if not records:
        return np.array([])

    features = [r.features for r in records]
    return np.array(features)
