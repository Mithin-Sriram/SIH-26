"""In-memory registry of thermal anomaly detections served by the API.

Sources:
  - a geographically seeded synthetic demo set (48 detections anchored to
    real Indian industrial / agricultural / forested regions) — always
    present
  - real FIRMS rows: loaded at startup from backend/app/data/firms.csv
    (pre-provided) or firms_last_upload.csv (previous upload), and added/
    replaced at runtime via POST /api/upload-firms

Every record is classified once on ingest (fast path, no SHAP); the full
detail bundle (evidence, why-not, SHAP) is computed lazily per id and
cached on the record.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from ..ml.explain import (
    investigation_priority,
    render_evidence,
    why_not_text,
)
from ..ml.predict import classify_detection, explain_all_classes
from .demo_seed import generate_seed
from .facilities import nearest_facility
from .firms import FirmsParseError, parse_firms_csv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
FIRMS_FILE = os.path.join(DATA_DIR, "firms.csv")
FIRMS_LAST_UPLOAD = os.path.join(DATA_DIR, "firms_last_upload.csv")

SEED = 26162
SYNTHETIC_BASE_DATE = datetime(2026, 1, 1, 4, 30, tzinfo=timezone.utc)

_records: list[dict[str, Any]] = []
_by_id: dict[str, dict[str, Any]] = {}
_syn_count = 0
_firms_count = 0


# --------------------------------------------------------------------------
# ingest helpers
# --------------------------------------------------------------------------

def _classify_record(record: dict[str, Any]) -> None:
    result = classify_detection(record["features"], with_shap=False)
    record["classification"] = result
    priority, reason = investigation_priority(
        result["predicted_class"], result["confidence"]
    )
    record["priority"] = priority
    record["priority_reason"] = reason
    record["detail"] = None


def _load_synthetic() -> None:
    global _syn_count
    for i, row in enumerate(generate_seed()):
        _syn_count += 1
        month = int(max(1, min(12, row["detection_month"])))
        detected_at = SYNTHETIC_BASE_DATE.replace(
            month=month, day=1 + (i % 28)
        )
        features = row["features"]
        record = {
            "id": f"syn-{_syn_count:04d}",
            "source": "synthetic",
            "features": features,
            "detected_at": detected_at,
            "measured_features": list(features.keys()),
            "raw": None,
        }
        _classify_record(record)
        _records.append(record)
        _by_id[record["id"]] = record


def _ingest_firms_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append FIRMS rows as new records; returns the new records."""
    global _firms_count
    added = []
    for r in rows:
        _firms_count += 1
        record = {
            "id": f"firms-{_firms_count:04d}",
            "source": "firms",
            "features": r["features"],
            "detected_at": r["detected_at"],
            "measured_features": list(r["features"].keys()),
            "raw": r.get("satellite"),
        }
        _classify_record(record)
        _records.append(record)
        _by_id[record["id"]] = record
        added.append(record)
    return added


def replace_firms_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace all FIRMS records with the given parsed rows."""
    global _firms_count
    _records[:] = [r for r in _records if r["source"] != "firms"]
    _by_id.clear()
    for r in _records:
        _by_id[r["id"]] = r
    _firms_count = 0
    return _ingest_firms_rows(rows)


def init_store(force: bool = False) -> None:
    """Build the registry: synthetic sample + any available FIRMS file."""
    if _records and not force:
        return
    _records.clear()
    _by_id.clear()
    _load_synthetic()
    path = None
    if os.path.exists(FIRMS_FILE):
        path = FIRMS_FILE
    elif os.path.exists(FIRMS_LAST_UPLOAD):
        path = FIRMS_LAST_UPLOAD
    if path:
        with open(path, "rb") as f:
            content = f.read()
        try:
            parsed = parse_firms_csv(content)
            _ingest_firms_rows(parsed.rows)
            print(f"loaded {len(parsed.rows)} FIRMS rows from {path}")
        except FirmsParseError as e:
            print(f"WARNING: could not load {path}: {e}")


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------

def get_records(source: Optional[str] = None,
                limit: int = 2000) -> list[dict[str, Any]]:
    recs = _records if source is None else \
        [r for r in _records if r["source"] == source]
    return recs[:limit]


def get_by_id(detection_id: str) -> Optional[dict[str, Any]]:
    return _by_id.get(detection_id)


def compute_stats() -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_priority = {"High": 0, "Medium": 0, "Low": 0}
    sources: dict[str, int] = {}
    for r in _records:
        cat = r["classification"]["predicted_class"]
        by_class[cat] = by_class.get(cat, 0) + 1
        by_priority[r["priority"]] += 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    return {
        "total": len(_records),
        "by_class": by_class,
        "by_priority": by_priority,
        "high_priority": by_priority["High"],
        "sources": sources,
    }


# --------------------------------------------------------------------------
# response builders
# --------------------------------------------------------------------------

def list_item(record: dict[str, Any]) -> dict[str, Any]:
    """Lean projection for the list endpoint (no raw feature dump)."""
    f = record["features"]
    c = record["classification"]
    note = (
        f"{c['predicted_class']} ({c['confidence']:.0%} confident); "
        f"I4 = {f.get('brightness_temp_i4_k', 0):.0f} K"
    )
    return {
        "id": record["id"],
        "latitude": round(float(f["latitude"]), 5),
        "longitude": round(float(f["longitude"]), 5),
        "frp_mw": round(float(f.get("frp_mw", 0.0)), 1),
        "confidence": round(float(f.get("confidence_pct", 0.0)) / 100.0, 3),
        "brightness_temp_k": round(float(f.get("brightness_temp_i4_k", 0.0)), 1),
        "detected_at": record["detected_at"],
        "source": record["source"],
        "category": c["predicted_class"],
        "category_probability": c["confidence"],
        "priority": record["priority"],
        "notes": note,
    }


def _build_detail(record: dict[str, Any]) -> dict[str, Any]:
    features = record["features"]
    full = classify_detection(features)  # with SHAP
    pred = full["predicted_class"]

    evidence = [
        render_evidence(c, pred)
        for c in full["shap_top_5"]
    ]

    why_not = None
    runner_up = full["top_3"][1] if len(full["top_3"]) > 1 else None
    if runner_up:
        shap_by_class = explain_all_classes(features)
        why_not = {
            "class_name": runner_up["class"],
            "probability": runner_up["probability"],
            "explanation": why_not_text(
                pred, runner_up["class"], runner_up["probability"],
                shap_by_class[pred], shap_by_class[runner_up["class"]],
                features,
            ),
        }

    f_lat = float(features.get("latitude", 0.0))
    f_lon = float(features.get("longitude", 0.0))
    nearest = nearest_facility(f_lat, f_lon)

    item = list_item(record)
    return {
        **item,
        "frp_mw": item["frp_mw"],
        "predicted_class": pred,
        "probability": full["confidence"],
        "priority_reason": record["priority_reason"],
        "top_3": full["top_3"],
        "evidence": evidence,
        "shap_top_5": full["shap_top_5"],
        "why_not": why_not,
        "nearest_industrial": nearest,
        "features": features,
        "measured_features": record["measured_features"],
        "model_version": full["model_version"],
    }


def ensure_detail(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("detail") is None:
        record["detail"] = _build_detail(record)
    return record["detail"]
