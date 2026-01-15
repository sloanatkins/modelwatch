import joblib
from pathlib import Path

MODEL_PATH = Path("models/saved_models/fraud_model_v1.pkl")

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, X):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

model_service = ModelService()
