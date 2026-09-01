"""Small gazetteer of major Indian refineries and industrial hubs.

Used to attach a real facility name + distance to each detection for the
detail endpoint ("nearest industrial facility, if available"). Coordinates
are approximate plant locations — fine for a demo-scale gazetteer.
"""

from __future__ import annotations

import math
from typing import Optional

FACILITIES: list[dict] = [
    # --- refineries / petrochemical ---
    {"name": "Jamnagar Refinery (Reliance)", "type": "refinery",
     "lat": 22.75, "lon": 69.95},
    {"name": "Koyali Refinery (IOCL Vadodara)", "type": "refinery",
     "lat": 22.28, "lon": 73.17},
    {"name": "Mathura Refinery (IOCL)", "type": "refinery",
     "lat": 27.48, "lon": 77.67},
    {"name": "Panipat Refinery (IOCL)", "type": "refinery",
     "lat": 29.39, "lon": 76.97},
    {"name": "Guru Gobind Singh Refinery (Bathinda)", "type": "refinery",
     "lat": 30.22, "lon": 74.95},
    {"name": "Barauni Refinery (IOCL)", "type": "refinery",
     "lat": 25.47, "lon": 86.03},
    {"name": "Haldia Refinery (IOCL)", "type": "refinery",
     "lat": 22.06, "lon": 88.11},
    {"name": "Bongaigaon Refinery (IOCL)", "type": "refinery",
     "lat": 26.49, "lon": 90.55},
    {"name": "Numaligarh Refinery (IOCL)", "type": "refinery",
     "lat": 26.62, "lon": 93.62},
    {"name": "Guwahati Refinery (IOCL)", "type": "refinery",
     "lat": 26.14, "lon": 91.75},
    {"name": "Kochi Refinery (BPCL)", "type": "refinery",
     "lat": 9.97, "lon": 76.28},
    {"name": "Chennai Refinery (CPCL Manali)", "type": "refinery",
     "lat": 13.17, "lon": 80.33},
    {"name": "Visakhapatnam Refinery (HPCL)", "type": "refinery",
     "lat": 17.72, "lon": 83.33},
    {"name": "Mumbai Refinery (BPCL Mahul)", "type": "refinery",
     "lat": 19.01, "lon": 72.86},
    {"name": "Mangalore Refinery (MRPL)", "type": "refinery",
     "lat": 12.92, "lon": 74.86},
    {"name": "Paradip Refinery (IOCL)", "type": "refinery",
     "lat": 20.32, "lon": 86.61},
    # --- steel / heavy industry ---
    {"name": "Bhilai Steel Plant (SAIL)", "type": "steel",
     "lat": 21.19, "lon": 81.35},
    {"name": "Jamshedpur Works (Tata Steel)", "type": "steel",
     "lat": 22.80, "lon": 86.20},
    {"name": "Rourkela Steel Plant (SAIL)", "type": "steel",
     "lat": 22.22, "lon": 84.86},
    {"name": "Bokaro Steel City (SAIL)", "type": "steel",
     "lat": 23.67, "lon": 86.15},
    # --- power / mining / metals ---
    {"name": "Korba Industrial Zone (NTPC / Balco)", "type": "power-metals",
     "lat": 22.35, "lon": 82.68},
    {"name": "Talcher Coalfields (MCL)", "type": "mining",
     "lat": 20.93, "lon": 85.13},
    {"name": "Singrauli Coal Belt (NCL)", "type": "mining",
     "lat": 24.12, "lon": 82.66},
    {"name": "Angul Aluminium Hub (NALCO)", "type": "metals",
     "lat": 20.84, "lon": 85.10},
    {"name": "Neyveli Lignite Complex (NLC)", "type": "power",
     "lat": 11.54, "lon": 79.48},
]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_facility(lat: float, lon: float,
                     max_distance_m: float = 300_000.0) -> Optional[dict]:
    """Nearest gazetteer facility within `max_distance_m`, or None."""
    best: Optional[dict] = None
    best_d = float("inf")
    for f in FACILITIES:
        d = haversine_m(lat, lon, f["lat"], f["lon"])
        if d < best_d:
            best_d = d
            best = f
    if best is None or best_d > max_distance_m:
        return None
    return {
        "name": best["name"],
        "facility_type": best["type"],
        "distance_m": int(round(best_d)),
    }
