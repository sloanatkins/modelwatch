# Database Layer

This module manages persistence for model predictions using SQLAlchemy and SQLite.

## Stored Data
Each prediction record includes:
- Timestamp
- Feature vector (JSON)
- Model prediction
- Predicted probability

## Why SQLite?
SQLite is used to:
- Keep the project self-contained
- Avoid external dependencies
- Mimic a lightweight production logging store

The schema is intentionally simple but extensible.
