from fastapi import FastAPI
from backend.api.predict import router as predict_router
from backend.database.session import engine, Base
from backend.api.drift import router as drift_router
from backend.database import models
990
app = FastAPI(
    title="ModelWatch",
    description="Real-time ML Model Monitoring & Drift Detection",
    version="1.0"
)

# Create database tables at startup
Base.metadata.create_all(bind=engine)

app.include_router(predict_router)
app.include_router(drift_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
