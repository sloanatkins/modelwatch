"""
Drift Threshold Definitions
===========================

This module defines how statistical drift metrics are translated
into actionable system states.

Raw metrics (KS, PSI) are difficult to interpret directly.
This file centralizes decision logic so drift severity is:
- Consistent
- Interpretable
- Easy to tune

Dependencies:
- None
"""

def classify_drift(ks_result: dict, psi_value: float):
    """
    Classify drift severity for a single feature.

    Returns:
    --------
    dict with:
    - status: healthy | warning | critical
    - reason: human-readable explanation
    """

    # Severe drift: statistically significant AND large population shift
    if ks_result["drift_detected"] and psi_value > 0.2:
        return {
            "status": "critical",
            "reason": "Statistically significant drift with severe population shift"
        }

    # Moderate drift: statistically significant but smaller shift
    if ks_result["drift_detected"] and psi_value > 0.1:
        return {
            "status": "warning",
            "reason": "Statistically significant drift with moderate population shift"
        }

    # No actionable drift
    return {
        "status": "healthy",
        "reason": "No significant or actionable drift detected"
    }
