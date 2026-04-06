"""
Drift API Endpoint
==================

This router exposes an endpoint to compute, persist, and return a drift report.

What it does:
-------------
1) Loads baseline feature distributions saved at training time
2) Loads a recent batch of prediction logs from the database
3) Runs drift tests (KS + PSI) feature-by-feature
4) Persists results to the drift_metrics table
5) Returns a clean JSON report

Key reliability upgrades:
-------------------------
- If baselines are missing -> returns a helpful 400 (not a 500)
- If there isn't enough logged data -> returns 200 with status + message
- All drift results are written to PostgreSQL for dashboard consumption
- Structured logging on every request

Dependencies:
- FastAPI
- NumPy
- SQLAlchemy
- logging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.database.crud import get_recent_predictions, log_drift_metrics
from backend.monitoring.baseline_loader import load_baselines
from backend.monitoring.statistical_tests import ks_test, psi_score
from backend.services.model_loader import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["Monitoring"])

# Alert thresholds (consistent with ADD)
KS_ALERT_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2


@router.get("/")
def get_drift_report(
    window_size: int = 500,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns drift metrics comparing the most recent `window_size` predictions
    vs baseline distributions. Results are persisted to drift_metrics table.

    Query params:
      window_size: how many most recent prediction rows to analyze
    """
    logger.info("Drift report requested | window_size=%d", window_size)

    # 1) Load baselines
    try:
        baselines = load_baselines()
    except FileNotFoundError as e:
        logger.error("Baseline load failed | error=%s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Load recent predictions from DB
    recent_rows = get_recent_predictions(db, limit=window_size)

    if len(recent_rows) < 50:
        logger.warning(
            "Not enough data for drift report | n_rows=%d", len(recent_rows)
        )
        return {
            "status": "not_enough_data",
            "message": (
                f"Need at least 50 logged predictions to compute drift. "
                f"Currently have {len(recent_rows)}. "
                f"Hit /predict a few more times or run the drift injector."
            ),
            "window_size": window_size,
            "n_rows": len(recent_rows),
            "features_checked": 0,
            "report": [],
        }

    try:
        X_current = np.array([r.features for r in recent_rows], dtype=float)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not parse features from prediction logs. "
                   "Make sure predictions are logged with a numeric features list."
        )

    # 3) Build drift report feature-by-feature
    feature_names: List[str] = list(baselines.keys())
    report = []

    n_current_features = X_current.shape[1]
    n_baseline_features = len(feature_names)
    n_compare = min(n_current_features, n_baseline_features)

    for i in range(n_compare):
        fname = feature_names[i]
        baseline_vals = baselines[fname]
        current_vals = X_current[:, i]

        ks = ks_test(baseline_vals, current_vals)
        psi = psi_score(baseline_vals, current_vals, bins=10)

        alert = ks["statistic"] > KS_ALERT_THRESHOLD or psi >= PSI_ALERT_THRESHOLD

        # 4) Persist to drift_metrics table
        log_drift_metrics(
            db=db,
            model_version=model_service.model_version,
            feature_name=fname,
            window_size=len(recent_rows),
            ks_statistic=ks["statistic"],
            ks_p_value=ks["p_value"],
            psi_score=psi,
            alert=alert,
        )

        report.append({
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
            "alert": alert,
        })

    # 5) Overall summary
    drifted_ks = sum(1 for r in report if r["ks_drifted"])
    significant_psi = sum(1 for r in report if r["psi"] >= PSI_ALERT_THRESHOLD)
    total_alerts = sum(1 for r in report if r["alert"])

    logger.info(
        "Drift report complete | features=%d ks_drifted=%d psi_significant=%d alerts=%d",
        n_compare, drifted_ks, significant_psi, total_alerts
    )

    return {
        "status": "ok",
        "window_size": window_size,
        "n_rows": len(recent_rows),
        "features_checked": n_compare,
        "summary": {
            "ks_drifted_features": drifted_ks,
            "psi_significant_features": significant_psi,
            "total_alerts": total_alerts,
        },
        "report": report,
    }