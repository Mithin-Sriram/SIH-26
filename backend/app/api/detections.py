"""API routes: detections, detail, stats, FIRMS upload, classify."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..data import detections_store as store
from ..data.firms import FirmsParseError, parse_firms_csv
from ..ml.predict import classify_detection
from ..models.schemas import (
    ClassificationResponse,
    DetectionDetail,
    DetectionListItem,
    ErrorResponse,
    StatsResponse,
    UploadResponse,
)

router = APIRouter()

FIRMS_HINT = (
    "Download a VIIRS 375m or MODIS CSV from "
    "https://firms.modaps.eosdis.nasa.gov/ — expected columns include "
    "latitude, longitude, bright_ti4 (or brightness), frp, confidence, "
    "acq_date, acq_time, daynight."
)


def _http_error(status: int, code: str, message: str,
                hint: Optional[str] = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail=ErrorResponse(error=code, message=message, hint=hint).model_dump(),
    )


@router.get("/detections", response_model=list[DetectionListItem])
def list_detections(
    source: Optional[str] = Query(
        None, description="Filter: 'synthetic', 'firms', or all"
    ),
    limit: int = Query(1000, ge=1, le=2000),
) -> list[dict[str, Any]]:
    """Lean list of detections with coordinates and classification."""
    return [store.list_item(r) for r in store.get_records(source, limit)]


@router.get("/stats", response_model=StatsResponse)
def stats() -> dict[str, Any]:
    """Counts by class / priority / source, for the dashboard header."""
    return store.compute_stats()


@router.get("/detections/{detection_id}", response_model=DetectionDetail)
def get_detection(detection_id: str) -> dict[str, Any]:
    """Full detail: raw features, top-3, evidence, why-not, priority."""
    record = store.get_by_id(detection_id)
    if record is None:
        raise _http_error(
            404, "not_found",
            f"detection '{detection_id}' does not exist.",
            "GET /api/detections for the list of valid ids.",
        )
    return store.ensure_detail(record)


@router.post("/upload-firms", response_model=UploadResponse)
async def upload_firms(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a FIRMS CSV; rows are classified and added to the store."""
    content = await file.read()
    if not content:
        raise _http_error(422, "invalid_firms_csv", "uploaded file is empty.",
                          FIRMS_HINT)

    try:
        parsed = parse_firms_csv(content)
    except FirmsParseError as e:
        raise _http_error(422, "invalid_firms_csv",
                          f"could not parse CSV: {e}", FIRMS_HINT) from e

    records = store.replace_firms_rows(parsed.rows)

    # persist the upload so a server restart keeps the demo data
    try:
        with open(store.FIRMS_LAST_UPLOAD, "wb") as f:
            f.write(content)
    except OSError:
        pass  # persistence is best-effort; in-memory store still updated

    return {
        "filename": file.filename or "upload.csv",
        "rows_received": parsed.received,
        "rows_classified": len(records),
        "rows_skipped": parsed.skipped,
        "skip_reasons": parsed.skip_reasons,
        "detections": [store.list_item(r) for r in records],
    }


@router.post("/classify", response_model=ClassificationResponse)
def classify(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify a raw feature dict (missing features use train defaults)."""
    if not isinstance(payload, dict) or not payload:
        raise _http_error(400, "bad_request",
                          "expected a non-empty JSON object of features.")
    try:
        return classify_detection(payload)
    except ValueError as e:
        raise _http_error(422, "invalid_features", str(e),
                          "See GET /api/detections/{id} for feature names.") \
            from e
