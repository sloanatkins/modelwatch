"""
ModelWatch Application Entry Point
====================================

Initializes the FastAPI application, configures structured logging,
creates database tables on startup, and registers all API routers.

Routers:
- /predict  — model inference endpoint
- /drift    — drift metrics endpoint
- /health   — health check

Dependencies:
- FastAPI
- SQLAlchemy
- logging
"""

import logging
from fastapi import FastAPI
from backend.api.predict import router as predict_router
from backend.api.drift import router as drift_router
from backend.database.session import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

app = FastAPI(
    title="ModelWatch",
    description="Real-time ML Model Monitoring & Drift Detection",
    version="1.0"
)

Base.metadata.create_all(bind=engine)

app.include_router(predict_router)
app.include_router(drift_router)


@app.get("/health")
def health_check():
    return {"status": "healthy"}
