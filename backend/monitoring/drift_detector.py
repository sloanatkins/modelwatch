"""
Drift Detection Engine
=====================

This module compares training-time baseline feature distributions
against recent production data and applies statistical tests and
decision thresholds to determine drift severity.

Dependencies:
- NumPy
- SQLAlchemy
"""

import numpy as np
from sqlalchemy.orm import Session

from backend.monitoring.windowing import get_recent_feature_window
from backend.monitoring.drift_metrics import ks_test, psi
from backend.monitoring.thresholds import classify_drift


def detect_feature_drift(
    db: Session,
    baseline_distributions: dict,
    window_size: int = 500
):
    """
    Detect and classify drift for each feature.

    Returns:
    --------
    dict : feature_index -> drift analysis
    """

    recent_data = get_recent_feature_window(db, window_size)

    if recent_data.size == 0:
        return {}

    drift_report = {}

    for feature_idx, baseline_values in baseline_distributions.items():
        current_values = recent_data[:, feature_idx]

        ks_result = ks_test(baseline_values, current_values)
        psi_value = psi(baseline_values, current_values)

        severity = classify_drift(ks_result, psi_value)

        drift_report[feature_idx] = {
            "ks": ks_result,
            "psi": psi_value,
            "severity": severity
        }

    return drift_report
