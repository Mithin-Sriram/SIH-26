"""FastAPI entrypoint for the SIH26162 thermal anomaly classifier.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.detections import router as detections_router
from .data import detections_store

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# For local single-server runs: serve the frontend build if it exists
# (frontend/ `npm run build`). On Render the backend is a pure API service.
STATIC_DIR = os.path.join(os.path.dirname(_BASE_DIR), "frontend", "dist")

# Comma-separated extra origins (e.g. your Vercel domain) via CORS_ORIGINS.
_origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
]
_origins += [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]


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
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sih26162-classifier"}


app.include_router(detections_router, prefix="/api", tags=["detections"])

# Serve the built frontend (frontend/ `npm run build` outputs to frontend/dist)
# so the whole app can run from a single origin locally. In dev and on
# Vercel/Render the frontend and API run separately (see README deployment).
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
