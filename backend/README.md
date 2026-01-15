Backend Service (FastAPI)
========================

This directory contains the **production monitoring service**.

The backend is responsible for:
- Loading trained ML models
- Serving real-time predictions
- Logging prediction data
- Detecting data drift and performance degradation
- Triggering alerts and retraining signals

---

Architecture Overview
---------------------
The backend is split into logical layers:

api/
- Defines HTTP endpoints (`/predict`, `/metrics`, etc.)

monitoring/
- Implements statistical drift detection
- Tracks model performance over time

database/
- Stores prediction history and metrics

services/
- Long-lived services such as model loading

utils/
- Shared helper functions and statistical tests

---

Important Design Rule
---------------------
The backend **never** reads raw training data.
It only interacts with model artifacts and live prediction traffic.
