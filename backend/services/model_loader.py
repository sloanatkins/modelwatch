"""
Model Loading Service
=====================

Loads the trained fraud detection model for inference.

Loading priority:
1. MLflow artifact store — loads by MODEL_RUN_ID env var if set
2. Joblib fallback — loads fraud_model_v1.pkl directly

The model_version attribute is set to the MLflow run ID when loaded
via MLflow, or 'v1-local' when loaded from the pickle fallback.
This value is logged with every prediction request.

Dependencies:
- MLflow
- joblib
- scikit-learn
"""

import os
import logging
import joblib
import mlflow.sklearn
from pathlib import Path

logger = logging.getLogger(__name__)

FALLBACK_MODEL_PATH = Path("models/saved_models/fraud_model_v1.pkl")


class ModelService:
    def __init__(self):
        self.model = None
        self.model_version = None
        self._load()

    def _load(self):
        run_id = os.getenv("MODEL_RUN_ID")

        if run_id:
            try:
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = run_id
                logger.info("Model loaded from MLflow | run_id=%s", run_id)
                return
            except Exception as e:
                logger.warning(
                    "MLflow load failed, falling back to pickle | error=%s", str(e)
                )

        # Fallback to pickle
        self.model = joblib.load(FALLBACK_MODEL_PATH)
        self.model_version = "v1-local"
        logger.info("Model loaded from pickle fallback | version=%s", self.model_version)

    def predict(self, X):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]


model_service = ModelService()
