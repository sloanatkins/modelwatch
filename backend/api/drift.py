"""
Drift API Endpoint
==================

This router exposes an endpoint to compute and return a drift report.

What it does:
-------------
1) Loads baseline feature distributions saved at training time
2) Loads a "current batch" of recent prediction logs from the database
3) Runs drift tests (KS + PSI) feature-by-feature
4) Returns a clean JSON report

Key reliability upgrades:
-------------------------
• If baselines are missing -> returns a helpful 400 (not a 500)
• If there isn't enough logged data -> returns 200 with status + message
• Works on Mac/Windows with stable repo-root pathing
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.database.crud import get_recent_predictions
from backend.monitoring.baseline_loader import load_baselines
from backend.monitoring.statistical_tests import ks_test, psi_score

router = APIRouter(prefix="/drift", tags=["Monitoring"])


@router.get("/")
def get_drift_report(
    window_size: int = 500,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns drift metrics comparing the most recent `window_size` predictions
    vs baseline distributions.

    Query params:
      window_size: how many most recent prediction rows to analyze
    """
    # 1) Load baselines (may raise FileNotFoundError -> convert to 400)
    try:
        baselines = load_baselines()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Load recent predictions from DB
    recent_rows = get_recent_predictions(db, limit=window_size)

    if len(recent_rows) < 50:
        return {
            "status": "not_enough_data",
            "message": (
                f"Need at least 50 logged predictions to compute drift. "
                f"Currently have {len(recent_rows)}. "
                f"Hit /predict a few more times (or run the simulator later)."
            ),
            "window_size": window_size,
            "n_rows": len(recent_rows),
            "features_checked": 0,
            "report": [],
        }

    # Each DB row should store a list/array of features in `features`
    # We'll stack them into a matrix of shape (n_samples, n_features)
    try:
        X_current = np.array([r.features for r in recent_rows], dtype=float)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not parse features from prediction logs. "
                   "Make sure predictions are being logged with a numeric `features` list."
        )

    # 3) Build drift report feature-by-feature
    feature_names: List[str] = list(baselines.keys())

    # NOTE: Feature order must match your model input order.
    # Your model uses columns from the dataset (V1..V28 + Amount) in that order.
    # If you saved baselines as individual files named 'V1', 'V2', ..., 'Amount',
    # this will naturally align IF you saved them consistently.
    report = []

    # If the number of baseline features doesn't match request features,
    # we can still compute drift for the overlap safely.
    n_current_features = X_current.shape[1]
    n_baseline_features = len(feature_names)
    n_compare = min(n_current_features, n_baseline_features)

    for i in range(n_compare):
        fname = feature_names[i]
        baseline_vals = baselines[fname]
        current_vals = X_current[:, i]

        ks = ks_test(baseline_vals, current_vals)
        psi = psi_score(baseline_vals, current_vals, bins=10)

        report.append(
            {
                "feature": fname,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "ks_drifted": ks["drift_detected"],
                "psi": psi,
                "psi_severity": (
                    "none" if psi < 0.1 else
                    "moderate" if psi < 0.2 else
                    "significant"
                ),
            }
        )

    # 4) Overall summary
    drifted_ks = sum(1 for r in report if r["ks_drifted"])
    significant_psi = sum(1 for r in report if r["psi"] >= 0.2)

    return {
        "status": "ok",
        "window_size": window_size,
        "n_rows": len(recent_rows),
        "features_checked": n_compare,
        "summary": {
            "ks_drifted_features": drifted_ks,
            "psi_significant_features": significant_psi,
        },
        "report": report,
    }
