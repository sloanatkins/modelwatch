# ModelWatch
### Production ML Model Observability & Drift Detection System


ModelWatch is a production-style machine learning monitoring system that serves predictions via an API, logs model behavior, and continuously checks for feature drift using statistical tests.

The goal of this project is to demonstrate how deployed ML models can be **observed, validated, and monitored over time**, not just trained once and forgotten.

---

## What This Project Demonstrates

- End-to-end ML lifecycle: **training → deployment → monitoring**
- Real-time prediction serving with FastAPI
- Persistent prediction logging using SQLite
- Feature distribution drift detection using:
  - Kolmogorov–Smirnov (KS) test
  - Population Stability Index (PSI)
- Sliding window analysis over recent predictions
- Baseline statistics generated from training data

This project is intentionally designed to resemble how **real ML systems are monitored in production**.

---


---

## 📦 Key Components

| Component | Description |
|--------|------------|
| `models/` | Training code and saved baseline artifacts |
| `backend/api/` | FastAPI endpoints |
| `backend/database/` | Prediction persistence layer |
| `backend/monitoring/` | Drift detection logic |
| `data/` | Raw and processed datasets |

---

## 🧪 API Endpoints

### `POST /predict`
- Accepts a feature vector
- Returns prediction + probability
- Logs the request for monitoring

### `GET /drift`
- Analyzes the most recent predictions
- Compares them to training baselines
- Returns drift metrics per feature

---

## ▶️ Running Locally

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload


