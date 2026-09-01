"""Parse NASA FIRMS fire-certificate CSV exports (VIIRS 375m or MODIS).

Normalises the varied FIRMS column names into our feature dict. Only the
thermal/geo features that FIRMS actually provides are filled in; the rest
are left missing and `predict.classify_detection` falls back to training
defaults for them.

Typical VIIRS columns:
    latitude, longitude, bright_ti4, scan, track, acq_date, acq_time,
    satellite, instrument, confidence, version, bright_ti5, frp, daynight
Typical MODIS columns:
    latitude, longitude, brightness, scan, track, acq_date, acq_time,
    satellite, instrument, confidence, version, bright_t31, frp, daynight
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

CONFIDENCE_WORD_MAP = {"l": 30.0, "n": 60.0, "h": 90.0,
                       "low": 30.0, "normal": 60.0, "high": 90.0}

COLUMN_ALIASES: dict[str, list[str]] = {
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long"],
    "frp_mw": ["frp", "frp_mw"],
    "brightness_temp_i4_k": ["bright_ti4", "brightness"],
    "brightness_temp_i5_k": ["bright_ti5", "bright_t31"],
    "confidence_pct": ["confidence", "confidence_pct"],
    "day_night_flag": ["daynight", "day_night", "daynight_flag"],
    "acq_date": ["acq_date", "date"],
    "acq_time": ["acq_time", "time"],
    "satellite": ["satellite"],
}

MAX_FIRMS_ROWS = 500


class FirmsParseError(Exception):
    """Raised when the uploaded file is not a usable FIRMS CSV."""


@dataclass
class ParsedFirms:
    rows: list[dict[str, Any]] = field(default_factory=list)
    received: int = 0
    skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        if len(self.skip_reasons) < 10:
            self.skip_reasons.append(reason)

    @property
    def first_reason(self) -> str:
        return self.skip_reasons[0] if self.skip_reasons else "unknown"


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    return None


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _parse_confidence(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in CONFIDENCE_WORD_MAP:
        return CONFIDENCE_WORD_MAP[s]
    return _to_float(v)


def _parse_timestamp(date_v: Any, time_v: Any) -> datetime | None:
    if date_v is None or (isinstance(date_v, float) and pd.isna(date_v)):
        return None
    try:
        d = pd.to_datetime(str(date_v).strip())
    except (ValueError, TypeError):
        return None
    hh, mm = 0, 0
    if time_v is not None and not (isinstance(time_v, float) and pd.isna(time_v)):
        try:
            t = str(int(float(time_v))).zfill(4)
            hh, mm = int(t[:2]), int(t[2:])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                hh, mm = 0, 0
        except (ValueError, TypeError):
            hh, mm = 0, 0
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc)


def parse_firms_csv(content: bytes) -> ParsedFirms:
    """Parse raw FIRMS CSV bytes into normalised rows.

    Raises FirmsParseError for undecodable / non-CSV / structurally
    invalid files. Rows with bad coordinates or timestamps are skipped
    (counted) rather than failing the whole upload.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError as e:
            raise FirmsParseError(f"could not decode file contents: {e}") from e

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as e:
        raise FirmsParseError("file contains no CSV data (empty).") from e
    except pd.errors.ParserError as e:
        raise FirmsParseError(f"file is not a well-formed CSV: {e}") from e

    if df.empty:
        raise FirmsParseError("CSV has a header but no data rows.")

    lat_col = _find_col(df, "latitude")
    lon_col = _find_col(df, "longitude")
    if lat_col is None or lon_col is None:
        raise FirmsParseError(
            "missing required 'latitude'/'longitude' columns. "
            f"Columns found: {', '.join(str(c) for c in list(df.columns)[:15])}"
        )

    cols = {k: _find_col(df, k) for k in COLUMN_ALIASES}
    parsed = ParsedFirms(received=len(df))

    for _, r in df.iterrows():
        if len(parsed.rows) >= MAX_FIRMS_ROWS:
            parsed._skip(f"row limit of {MAX_FIRMS_ROWS} reached")
            break
        lat = _to_float(r[lat_col])
        lon = _to_float(r[lon_col])
        if lat is None or lon is None:
            parsed._skip("non-numeric latitude/longitude")
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            parsed._skip(f"out-of-range coordinates ({lat}, {lon})")
            continue

        detected_at = _parse_timestamp(
            r[cols["acq_date"]] if cols["acq_date"] else None,
            r[cols["acq_time"]] if cols["acq_time"] else None,
        )
        if detected_at is None:
            parsed._skip("missing or invalid acq_date")
            continue

        features: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "detection_month": detected_at.month,
        }

        frp = _to_float(r[cols["frp_mw"]]) if cols["frp_mw"] else None
        if frp is not None and frp >= 0:
            features["frp_mw"] = frp

        i4 = _to_float(r[cols["brightness_temp_i4_k"]]) if cols["brightness_temp_i4_k"] else None
        if i4 is not None and i4 > 0:
            features["brightness_temp_i4_k"] = i4

        i5 = _to_float(r[cols["brightness_temp_i5_k"]]) if cols["brightness_temp_i5_k"] else None
        if i5 is not None and i5 > 0:
            features["brightness_temp_i5_k"] = i5

        conf = _parse_confidence(
            r[cols["confidence_pct"]] if cols["confidence_pct"] else None
        )
        if conf is not None:
            features["confidence_pct"] = float(min(max(conf, 0.0), 100.0))

        if cols["day_night_flag"]:
            dn = str(r[cols["day_night_flag"]]).strip().upper()
            if dn.startswith("N"):
                features["day_night_flag"] = "night"
            elif dn.startswith("D"):
                features["day_night_flag"] = "day"

        satellite = None
        if cols["satellite"]:
            sv = r[cols["satellite"]]
            if sv is not None and not (isinstance(sv, float) and pd.isna(sv)):
                satellite = str(sv)

        parsed.rows.append({
            "features": features,
            "detected_at": detected_at,
            "satellite": satellite or "FIRMS",
        })

    if not parsed.rows:
        raise FirmsParseError(
            f"no valid rows could be parsed (first problem: "
            f"{parsed.first_reason})."
        )
    return parsed
