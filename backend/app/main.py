"""FastAPI entrypoint for the SIH26162 thermal anomaly classifier.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.detections import router as detections_router
from .data import detections_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detections_store.init_store()
    except FileNotFoundError as e:
        print(f"WARNING: detection store not initialised: {e}")
    yield


app = FastAPI(
    title="SIH26162 Thermal Anomaly Classifier",
    description=(
        "Classifies satellite-detected thermal anomalies into Industrial "
        "Fire, Gas Flare, Wildfire, Agricultural Burning, or Other/Unknown. "
        "XGBoost + Optuna + Platt calibration + SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sih26162-classifier"}


app.include_router(detections_router, prefix="/api", tags=["detections"])
