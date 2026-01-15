# API Layer

This module defines the external interface of the system.

## Endpoints

### `POST /predict`
- Runs model inference
- Returns:
  - Binary prediction
  - Fraud probability
- Logs the following to the database:
  - Feature vector
  - Prediction
  - Probability
  - Timestamp

### `GET /drift`
- Performs drift detection on recent predictions
- Uses a sliding window of the most recent N rows
- Compares feature distributions against training baselines

This separation allows inference and monitoring to evolve independently.
