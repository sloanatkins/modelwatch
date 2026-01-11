"""
Statistical Drift Tests
======================

This module implements the core statistics behind drift detection.

Included tests:
---------------
1) KS Test (Kolmogorov–Smirnov)
   - Detects whether two distributions are different
   - Output: statistic + p-value

2) PSI (Population Stability Index)
   - Measures how significant the drift is
   - PSI thresholds:
       < 0.10 -> no meaningful drift
       0.10–0.20 -> moderate drift
       > 0.20 -> significant drift
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import ks_2samp


def ks_test(baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """
    Runs a two-sample KS test.
    """
    stat, p_value = ks_2samp(baseline, current)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
    }


def psi_score(baseline: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Computes Population Stability Index (PSI).

    PSI = sum( (cur_i - base_i) * ln(cur_i / base_i) )
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Use baseline quantiles as bin edges (stable for different ranges)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(baseline, quantiles)

    # Avoid duplicate edges (happens with constant-ish features)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    base_counts, _ = np.histogram(baseline, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    base_pct = base_counts / max(base_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    base_pct = np.clip(base_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
    return float(psi)
