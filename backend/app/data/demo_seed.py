"""Geographically realistic demo seed for the map.

Generates ~48 detections anchored to real Indian industrial / agricultural
/ forested regions instead of scattered random points:

  - Gas Flare            -> Gujarat refineries (Jamnagar, Koyali, Hazira)
  - Industrial Fire      -> Odisha-Chhattisgarh mining/steel belt
                            (Talcher, Angul, Korba, Singrauli)
  - Wildfire             -> Uttarakhand + Himachal forest ranges
  - Agricultural Burning -> Punjab + Haryana (stubble-burning belt)
  - Other/Unknown        -> scattered major cities

Feature values are sampled from the class profiles in
generate_synthetic_data (so each detection is internally consistent); the
final class shown on the map is whatever the trained model predicts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .generate_synthetic_data import PROFILES, _sample_profile

_SEED = 26162

ANCHORS: dict[str, list[tuple[float, float]]] = {
    "Gas Flare": [
        (22.75, 69.95), (22.73, 70.02), (22.79, 69.88),
        (22.68, 70.05), (22.82, 69.91),
        (22.28, 73.17), (22.33, 73.11),
        (21.10, 72.65),
    ],
    "Industrial Fire": [
        (20.93, 85.13), (20.90, 85.20), (20.98, 85.05),
        (20.84, 85.10), (20.80, 85.16),
        (22.35, 82.68), (22.30, 82.74),
        (24.12, 82.66),
    ],
    "Wildfire": [
        (30.29, 78.03), (29.85, 79.10), (30.35, 79.45), (29.60, 80.00),
        (30.70, 78.60), (30.05, 79.70), (30.55, 79.15),
        (31.90, 77.10), (32.20, 76.90), (31.60, 77.90),
        (32.50, 76.80), (31.10, 78.20),
    ],
    "Agricultural Burning": [
        (31.63, 74.87), (30.90, 75.85), (30.34, 76.38), (30.21, 74.94),
        (30.24, 75.84), (31.33, 75.58), (30.92, 74.61), (30.80, 75.17),
        (29.69, 76.99), (29.15, 75.72), (29.80, 76.40), (30.00, 76.88),
        (29.53, 75.03), (28.90, 76.57),
    ],
    "Other/Unknown": [
        (28.61, 77.21), (19.08, 72.88), (13.08, 80.27),
        (22.98, 88.43), (26.85, 80.95), (23.03, 72.58),
    ],
}


def _py(v):
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def generate_seed() -> list[dict]:
    """Return seeded detections: [{features: {...}, detection_month: int}]"""
    rng = np.random.default_rng(_SEED)
    rows: list[dict] = []
    for cls, anchors in ANCHORS.items():
        profile = replace(PROFILES[cls], n=len(anchors))
        df = _sample_profile(profile, rng)
        for i, (_, r) in enumerate(df.iterrows()):
            lat0, lon0 = anchors[i]
            features = {
                k: _py(v) for k, v in r.items() if k != "class_label"
            }
            features["latitude"] = round(float(lat0 + rng.normal(0, 0.05)), 5)
            features["longitude"] = round(float(lon0 + rng.normal(0, 0.05)), 5)
            rows.append({
                "features": features,
                "detection_month": int(r["detection_month"]),
            })
    return rows
