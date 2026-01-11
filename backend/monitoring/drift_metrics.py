"""
Statistical Drift Metrics
========================

This module implements statistical tests used to detect data drift
between training-time baseline distributions and live production data.

Implemented metrics:
- Kolmogorov–Smirnov (KS) Test
- Population Stability Index (PSI)

These metrics are computed feature-by-feature and form the core
signal for drift detection in ModelWatch.

Dependencies:
- NumPy
- SciPy
"""

import numpy as np
from scipy.stats import ks_2samp


def ks_test(baseline: np.ndarray, current: np.ndarray):
    """
    Perform Kolmogorov–Smirnov test between two distributions.

    Returns:
    --------
    dict with:
    - statistic
    - p_value
    - drift_detected (bool)
    """

    statistic, p_value = ks_2samp(baseline, current)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": p_value < 0.05
    }


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10):
    """
    Compute Population Stability Index (PSI).

    PSI interpretation:
    - < 0.1  : no significant drift
    - 0.1–0.2: moderate drift
    - > 0.2  : severe drift
    """

    baseline = np.asarray(baseline)
    current = np.asarray(current)

    # Create bins from baseline
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(baseline, quantiles)

    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    # Avoid division by zero
    baseline_pct = np.clip(baseline_pct, 1e-6, None)
    current_pct = np.clip(current_pct, 1e-6, None)

    psi_value = np.sum(
        (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    )

    return float(psi_value)
