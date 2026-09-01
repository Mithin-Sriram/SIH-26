"""Pydantic schemas for the SIH26162 API responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class FireCategory(str, Enum):
    INDUSTRIAL_FIRE = "Industrial Fire"
    GAS_FLARE = "Gas Flare"
    WILDFIRE = "Wildfire"
    AGRICULTURAL_BURNING = "Agricultural Burning"
    OTHER_UNKNOWN = "Other/Unknown"


# --- classification (POST /classify and detail) --------------------------

class ClassProbability(BaseModel):
    """Class + calibrated probability. Serialised as {"class": ..., } to
    match predict.classify_detection's output shape."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class", description="Class label")
    probability: float = Field(..., ge=0.0, le=1.0)


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: list[ClassProbability]
    shap_top_5: list[ShapContribution]
    model_version: str


# --- detection list -------------------------------------------------------

class DetectionListItem(BaseModel):
    """Lean projection — no raw 25-feature dump."""

    id: str
    latitude: float
    longitude: float
    frp_mw: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    brightness_temp_k: float
    detected_at: datetime
    source: str
    category: str
    category_probability: float = Field(..., ge=0.0, le=1.0)
    priority: str
    notes: Optional[str] = None


# --- detection detail -----------------------------------------------------

class WhyNot(BaseModel):
    class_name: str
    probability: float
    explanation: str


class NearestFacility(BaseModel):
    name: str
    facility_type: str
    distance_m: int


class DetectionDetail(DetectionListItem):
    predicted_class: str
    probability: float
    priority_reason: str
    top_3: list[ClassProbability]
    evidence: list[str]
    shap_top_5: list[ShapContribution]
    why_not: Optional[WhyNot] = None
    nearest_industrial: Optional[NearestFacility] = None
    features: dict[str, Any]
    measured_features: list[str]
    model_version: str


# --- stats ----------------------------------------------------------------

class StatsResponse(BaseModel):
    total: int
    by_class: dict[str, int]
    by_priority: dict[str, int]
    high_priority: int
    sources: dict[str, int]


# --- FIRMS upload ---------------------------------------------------------

class UploadResponse(BaseModel):
    filename: str
    rows_received: int
    rows_classified: int
    rows_skipped: int
    skip_reasons: list[str] = []
    detections: list[DetectionListItem]


class ErrorResponse(BaseModel):
    error: str
    message: str
    hint: Optional[str] = None
