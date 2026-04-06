"""
Fraud Detection Model Training & Baseline Generation
====================================================

This script trains a RandomForest-based fraud detection model using the
Credit Card Transactions dataset and generates all artifacts required
for production inference and drift monitoring.

Artifacts Produced:
-------------------
1. Trained model logged to MLflow (also saved as joblib fallback)
   - Used by the FastAPI /predict endpoint

2. Baseline performance & metadata (JSON)
   - Used for documentation, dashboards, and monitoring context

3. Baseline feature distributions (NumPy arrays)
   - Used by the drift detection system (KS test, PSI)
   - Stored per-feature using PRE-SMOTE training data

IMPORTANT DESIGN NOTES:
-----------------------
- SMOTE is applied ONLY for model training
- Drift baselines MUST reflect real production distributions
- Therefore, baselines are computed from X_train BEFORE SMOTE
- MLflow run ID is printed at the end — set it as MODEL_RUN_ID env var
"""

# =========================
# Imports
# =========================
import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


# =========================
# Paths & Directories
# =========================
RAW_DATA_PATH = "data/raw/creditcard.csv"

MODEL_DIR = "models/saved_models"
BASELINE_DIR = "artifacts/baseline"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)


# =========================
# Load Dataset
# =========================
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

X = df.drop(columns=["Class", "Time"])
y = df["Class"]


# =========================
# Train / Test Split
# =========================
print("Splitting dataset (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Handle Class Imbalance (Training Only)
# =========================
print("Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


# =========================
# Train Model
# =========================
print("Training RandomForest model...")
params = {
    "n_estimators": 150,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": 42
}

model = RandomForestClassifier(**params)
model.fit(X_train_bal, y_train_bal)


# =========================
# Evaluate Model
# =========================
print("Evaluating model on untouched test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": roc_auc_score(y_test, y_pred_proba),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred)
}

print("\nBaseline Performance:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")


# =========================
# Log to MLflow
# =========================
print("\nLogging to MLflow...")
mlflow.set_experiment("modelwatch-fraud-detection")

with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")
    print("Set this as your MODEL_RUN_ID environment variable.")


# =========================
# Save Model (joblib fallback)
# =========================
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_v1.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")


# =========================
# Save Baseline Metadata
# =========================
baseline_stats = {
    "mlflow_run_id": run_id,
    "n_samples": int(len(X_train)),
    "class_distribution": {int(k): int(v) for k, v in y_train.value_counts().items()},
    "feature_means": X_train.mean().to_dict(),
    "feature_stds": X_train.std().to_dict(),
    "performance": metrics
}

BASELINE_STATS_PATH = os.path.join(MODEL_DIR, "baseline_stats_v1.json")
with open(BASELINE_STATS_PATH, "w") as f:
    json.dump(baseline_stats, f, indent=2)

print(f"Baseline stats saved to: {BASELINE_STATS_PATH}")


# =========================
# Save Baseline Distributions (FOR DRIFT DETECTION)
# =========================
print("\nSaving baseline feature distributions for drift detection...")

for col in X_train.columns:
    np.save(
        os.path.join(BASELINE_DIR, f"{col}.npy"),
        X_train[col].values
    )

print(f"Baseline distributions saved to: {BASELINE_DIR}")
print("\nTraining & baseline generation complete.")