"""
Drift Injection Script
=======================

Simulates a prediction stream with controlled drift injection.
Sends batches of POST /predict requests in three phases:

Phase 1 (baseline): Feature distributions match training data — no drift.
Phase 2 (gradual):  Features shift progressively across batches.
Phase 3 (significant): Distributions are meaningfully shifted — alerts should fire.

Usage:
    python scripts/inject_drift.py
    python scripts/inject_drift.py --batches 30 --drift-start 10 --requests-per-batch 20

Dependencies:
- requests
- numpy
"""

import argparse
import time
import logging
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"

# Credit card dataset feature order: V1-V28 + Amount
N_FEATURES = 29

# Baseline feature stats (approximate means/stds from training data)
# V1-V28 are PCA components centered near 0, Amount has a different scale
BASELINE_MEANS = [0.0] * 28 + [88.0]   # Amount mean ~$88
BASELINE_STDS = [1.5] * 28 + [250.0]  # Amount std ~$250


def generate_features(drift_factor: float = 0.0) -> list:
    """
    Generates a single feature vector.
    drift_factor=0.0 -> baseline distribution
    drift_factor=1.0 -> fully shifted distribution
    """
    features = []
    for i in range(N_FEATURES):
        mean = BASELINE_MEANS[i] + drift_factor * 3.0 * BASELINE_STDS[i]
        std = BASELINE_STDS[i]
        val = float(np.random.normal(mean, std))
        features.append(round(val, 6))
    return features


def send_prediction(features: list) -> bool:
    try:
        response = requests.post(
            API_URL,
            json={"features": features},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        logger.error("Request failed | error=%s", str(e))
        return False


def run_injection(batches: int, drift_start: int, requests_per_batch: int):
    logger.info(
        "Starting drift injection | batches=%d drift_start=%d requests_per_batch=%d",
        batches, drift_start, requests_per_batch
    )

    total_sent = 0
    total_failed = 0

    for batch in range(1, batches + 1):
        # Calculate drift factor for this batch
        if batch < drift_start:
            # Phase 1: clean baseline
            drift_factor = 0.0
            phase = "baseline"
        elif batch < drift_start + (batches - drift_start) // 2:
            # Phase 2: gradual drift
            progress = (batch - drift_start) / ((batches - drift_start) / 2)
            drift_factor = progress * 0.5
            phase = "gradual"
        else:
            # Phase 3: significant drift
            drift_factor = 1.0
            phase = "significant"

        batch_success = 0
        for _ in range(requests_per_batch):
            features = generate_features(drift_factor)
            if send_prediction(features):
                batch_success += 1
            else:
                total_failed += 1

        total_sent += batch_success
        logger.info(
            "Batch %d/%d | phase=%-11s drift_factor=%.2f | sent=%d",
            batch, batches, phase, drift_factor, batch_success
        )

        # Small delay between batches
        time.sleep(0.1)

    logger.info(
        "Injection complete | total_sent=%d total_failed=%d",
        total_sent, total_failed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject simulated predictions with drift")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--drift-start", type=int, default=10)
    parser.add_argument("--requests-per-batch", type=int, default=20)
    args = parser.parse_args()

    run_injection(
        batches=args.batches,
        drift_start=args.drift_start,
        requests_per_batch=args.requests_per_batch,
    )
