# Fraud Detection Model — Baseline v1

## Overview
Binary classification model for detecting fraudulent credit card transactions.

This model serves as the **production baseline** for ModelWatch and provides
the reference distributions and performance metrics used for ongoing
data drift and model health monitoring.

The emphasis is not on maximizing accuracy, but on establishing a **stable,
interpretable baseline** suitable for production monitoring.

---

## Dataset
- **Source:** ULB Credit Card Fraud Dataset
- **Samples:** 284,807 transactions
- **Fraud Rate:** ~0.17%
- **Features:** 28 PCA-transformed numerical features + Amount
- **Target:** Class (0 = normal, 1 = fraud)

---

## Model
- **Algorithm:** Random Forest Classifier
- **Training Strategy:** SMOTE for class imbalance
- **Random State:** 42

---

## Baseline Performance
| Metric     | Value |
|-----------|-------|
| ROC-AUC   | ~0.98 |
| Precision | ~0.90 |
| Recall    | ~0.85 |

These metrics are recorded as reference values and can be used to detect
performance degradation over time.

---

## Saved Baseline Artifacts

The following artifacts are generated during training and consumed by the
monitoring system:

- Trained model (`.pkl`)
- Feature distributions (NumPy arrays)
- Feature summary statistics
- Class distribution
- Baseline performance metrics

These artifacts allow ModelWatch to compare **live prediction data**
against the original training distribution.

---

## Drift Monitoring Setup

Baseline statistics are used to compute drift using:

- Kolmogorov–Smirnov Test (distribution shift detection)
- Population Stability Index (magnitude of change)
- Jensen–Shannon Divergence (distribution similarity)

### Alert Thresholds
- PSI > 0.2 → Significant feature drift
- ROC-AUC drop > 5% → Potential performance degradation

---

## Purpose

This model is intentionally simple.

The objective is not state-of-the-art accuracy, but to demonstrate how a
deployed ML model can be **observed, evaluated, and monitored over time**
in a production-style system.
