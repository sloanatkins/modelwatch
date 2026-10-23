# the real model (fraud_model_v1.pkl) isn't in the repo, so CI doesn't have it.
# if it's missing, train a tiny stand-in model on random data so the API can still load.
# locally the real model is there, so this does nothing.

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path("models/saved_models/fraud_model_v1.pkl")
N_FEATURES = 29  # V1-V28 plus Amount, same as the real model


def pytest_configure(config):
    if MODEL_PATH.exists():
        return

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, N_FEATURES))
    y = rng.integers(0, 2, size=200)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
