# ModelWatch
### Production ML Observability & Drift Detection System

ModelWatch is a production-style ML monitoring system built around a core question: what happens to a deployed model after it ships? It serves predictions through a FastAPI API, logs every inference to PostgreSQL, runs statistical drift detection on a schedule, and surfaces everything through a Streamlit dashboard.

The system is designed to resemble how real ML teams monitor models in production — not just train them once and call it done.

## Dashboard

![ModelWatch Top](assets/dashboard_top.png)
![ModelWatch Drift](assets/dashboard_drift.png)
![ModelWatch Alerts](assets/dashboard_alerts.png)
---

## What This Demonstrates

- End-to-end ML lifecycle: **training → deployment → monitoring**
- Prediction serving with FastAPI and structured logging on every request
- Persistent prediction logging to PostgreSQL (timestamp, model version, features, latency, status)
- Feature drift detection using KS test and PSI
- Automated monitoring engine that runs drift detection on a configurable interval
- MLflow experiment tracking across model versions
- Streamlit + Plotly monitoring dashboard
- Full Docker stack — all four services spin up with one command
- GitHub Actions CI — lint and test on every push

---

## Architecture

| Component | Tool |
|-----------|------|
| Prediction API | FastAPI + Uvicorn |
| Database | PostgreSQL |
| Model Registry | MLflow |
| Drift Detection | KS test + PSI (scipy + numpy) |
| Monitoring Engine | Python scheduler |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Monitoring Questions Answered

- **Is my model's input distribution today different from what it was trained on?** KS statistic and PSI score per feature, compared against training baselines.
- **Which features are drifting and by how much?** Per-feature breakdown sorted by KS statistic, with alert threshold lines.
- **How has prediction confidence changed over time?** Fraud probability distribution across the full prediction window.
- **When did drift start?** Prediction volume chart shows injection phases — baseline, gradual, significant.
- **What is my model's latency profile?** Per-request latency over time, surfaced from the prediction log.

---

## Project Structure

```
modelwatch/
├── backend/
│   ├── api/          # FastAPI endpoints (predict, drift)
│   ├── database/     # PostgreSQL models, session, CRUD
│   ├── monitoring/   # KS test, PSI, scheduler, baseline loader
│   └── services/     # MLflow model loader
├── dashboard/        # Streamlit + Plotly dashboard
├── models/           # Training script, saved artifacts
├── scripts/          # Drift injection simulator
├── tests/            # pytest smoke tests
├── Dockerfile.api
├── Dockerfile.dashboard
└── docker-compose.yml
```

---

## Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it at `data/raw/creditcard.csv`.

---

## Setup

### Train the model

```bash
pip install -r requirements.txt
python models/train_model.py
```

Copy the printed MLflow run ID — you need it as an environment variable.

### Run with Docker

```bash
export MODEL_RUN_ID=<your_mlflow_run_id>
docker-compose up --build
```

Services available at:
- API: http://localhost:8000 — OpenAPI docs at /docs
- Dashboard: http://localhost:8501

### Run locally without Docker

```bash
brew services start postgresql@16
psql postgres -c "CREATE USER modelwatch WITH PASSWORD 'modelwatch';"
psql postgres -c "CREATE DATABASE modelwatch OWNER modelwatch;"

export DATABASE_URL=postgresql://modelwatch:modelwatch@localhost:5432/modelwatch
export MODEL_RUN_ID=<your_mlflow_run_id>

uvicorn backend.main:app --reload
python -m backend.monitoring.scheduler
streamlit run dashboard/app.py
```

### Inject a simulated drift stream

```bash
python scripts/inject_drift.py --batches 30 --drift-start 10 --requests-per-batch 20
```

Sends 600 predictions across three phases — baseline, gradual drift, significant drift.

---

## API Endpoints

- `POST /predict` — accepts a feature vector, returns prediction and fraud probability, logs full request to PostgreSQL
- `GET /drift/` — computes KS + PSI per feature vs training baselines, writes results to drift_metrics table
- `GET /health` — returns `{"status": "healthy"}`

---

## CI

![CI](https://github.com/sloanatkins/modelwatch/actions/workflows/ci.yml/badge.svg)

GitHub Actions runs on every push to main: installs dependencies, lints with flake8, runs pytest.

---

Built with Python, FastAPI, PostgreSQL, MLflow, Streamlit, Plotly, Docker, and GitHub Actions.