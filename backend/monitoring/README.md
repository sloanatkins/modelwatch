# Drift Monitoring System

This module implements feature drift detection by comparing recent prediction data against training baselines.

## Baseline Generation
Baselines are created during training and include:
- Feature means and standard deviations
- Full feature distributions (NumPy arrays)

## Drift Detection Process

1. Load training baseline distributions
2. Fetch recent predictions from the database
3. Apply sliding window selection
4. For each feature:
   - KS test (distribution shift)
   - PSI (magnitude of shift)
5. Flag features exceeding thresholds

## Why KS + PSI?

- KS detects whether distributions differ
- PSI quantifies how significant the change is

Using both prevents false positives and improves robustness.

## Output
The drift endpoint returns:
- Features analyzed
- Drift metrics per feature
- High-level status indicators
