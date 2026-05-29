"""
Monitoring Engine / Scheduler
==============================

Runs drift detection on a configurable interval.

On each cycle:
1. Pulls the last N predictions from prediction_log
2. Loads baseline feature distributions
3. Computes KS statistic and PSI score per feature
4. Writes results to drift_metrics table
5. Logs a summary of alerts

Run this as a standalone process alongside the API:
  python -m backend.monitoring.scheduler

Dependencies:
- SQLAlchemy
- NumPy
- SciPy
- logging
"""

import time
import logging
import os
import numpy as np

from backend.database.session import SessionLocal
from backend.database.crud import get_recent_predictions, log_drift_metrics
from backend.monitoring.baseline_loader import load_baselines
from backend.monitoring.statistical_tests import ks_test, psi_score
from backend.services.model_loader import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Configurable via env vars
INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", 300))  # default 5 min
WINDOW_SIZE = int(os.getenv("MONITOR_WINDOW_SIZE", 500))
KS_ALERT_THRESHOLD = float(os.getenv("KS_ALERT_THRESHOLD", 0.1))
PSI_ALERT_THRESHOLD = float(os.getenv("PSI_ALERT_THRESHOLD", 0.2))


def run_monitoring_cycle():
    """
    Runs a single drift detection cycle.
    Pulls recent predictions, computes drift per feature, persists results.
    """
    logger.info("Starting monitoring cycle | window_size=%d", WINDOW_SIZE)

    db = SessionLocal()
    try:
        # Load baselines
        try:
            baselines = load_baselines()
        except FileNotFoundError as e:
            logger.error("Baseline load failed — skipping cycle | error=%s", str(e))
            return

        # Pull recent predictions
        recent_rows = get_recent_predictions(db, limit=WINDOW_SIZE)

        if len(recent_rows) < 50:
            logger.warning(
                "Not enough predictions for drift analysis | n_rows=%d (need 50)",
                len(recent_rows)
            )
            return

        try:
            X_current = np.array([r.features for r in recent_rows], dtype=float)
        except Exception as e:
            logger.error("Failed to parse features from prediction_log | error=%s", str(e))
            return

        feature_names = list(baselines.keys())
        n_compare = min(X_current.shape[1], len(feature_names))

        alerts = []

        for i in range(n_compare):
            fname = feature_names[i]
            baseline_vals = baselines[fname]
            current_vals = X_current[:, i]

            ks = ks_test(baseline_vals, current_vals)
            psi = psi_score(baseline_vals, current_vals, bins=10)

            alert = ks["statistic"] > KS_ALERT_THRESHOLD or psi >= PSI_ALERT_THRESHOLD

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

            if alert:
                alerts.append(fname)

        logger.info(
            "Monitoring cycle complete | features=%d alerts=%d alert_features=%s",
            n_compare, len(alerts), alerts if alerts else "none"
        )

    finally:
        db.close()


def main():
    logger.info(
        "Monitoring engine started | interval=%ds window=%d",
        INTERVAL_SECONDS, WINDOW_SIZE
    )
    while True:
        try:
            run_monitoring_cycle()
        except Exception as e:
            logger.error("Monitoring cycle failed unexpectedly | error=%s", str(e))
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
