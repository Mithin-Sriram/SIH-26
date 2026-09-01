"""Human-readable explanations for classifications.

Turns raw SHAP feature contributions into strings a judge can read, e.g.
"High thermal intensity (FRP = 48 MW) — supports Wildfire", plus the
"why not the runner-up class" text and the investigation-priority rule.
"""

from __future__ import annotations

from typing import Any

# Threshold-banded description templates per feature. Each band is
# (upper_bound, text) — the first band whose bound exceeds the value wins.
BANDS: dict[str, list[tuple[float, str]]] = {
    "frp_mw": [
        (20, "Low fire radiative power (FRP = {v:.0f} MW)"),
        (150, "Moderate fire radiative power (FRP = {v:.0f} MW)"),
        (400, "High fire radiative power (FRP = {v:.0f} MW)"),
        (float("inf"), "Extreme fire radiative power (FRP = {v:.0f} MW)"),
    ],
    "brightness_temp_i4_k": [
        (360, "Cool I4 thermal band ({v:.0f} K — near background)"),
        (700, "Warm I4 thermal band ({v:.0f} K)"),
        (1600, "Hot I4 thermal band ({v:.0f} K)"),
        (float("inf"),
         "Extremely hot I4 thermal band ({v:.0f} K — well above the "
         "~1600 K gas-flare threshold)"),
    ],
    "brightness_temp_i5_k": [
        (335, "Near-background I5 band ({v:.0f} K)"),
        (float("inf"), "Elevated I5 thermal band ({v:.0f} K)"),
    ],
    "confidence_pct": [
        (50, "Low detection confidence ({v:.0f}%)"),
        (75, "Moderate detection confidence ({v:.0f}%)"),
        (float("inf"), "High detection confidence ({v:.0f}%)"),
    ],
    "persistence_days": [
        (10, "Short-lived hotspot ({v:.0f} days)"),
        (60, "Persisting hotspot ({v:.0f} days)"),
        (100, "Long-persisting hotspot ({v:.0f} days)"),
        (float("inf"),
         "Near-permanent hotspot ({v:.0f} days — flare-like persistence)"),
    ],
    "total_detections_365d": [
        (10, "Few detections in the past year ({v:.0f})"),
        (50, "Recurring detections in the past year ({v:.0f})"),
        (float("inf"), "Detected on most overpasses ({v:.0f}/yr)"),
    ],
    "frp_std_dev_pct": [
        (20, "Very steady heat output (FRP variability {v:.0f}%)"),
        (50, "Fluctuating heat output (FRP variability {v:.0f}%)"),
        (float("inf"), "Highly erratic heat output (FRP variability {v:.0f}%)"),
    ],
    "day_night_ratio": [
        (0.4, "Mostly daytime detections (night ratio {v:.2f})"),
        (0.6, "Mixed day/night detections (night ratio {v:.2f})"),
        (float("inf"), "Mostly night-time detections (night ratio {v:.2f})"),
    ],
    "active_days_count": [
        (30, "Burning on few days ({v:.0f} active days)"),
        (100, "Burning on many days ({v:.0f} active days)"),
        (float("inf"), "Active nearly year-round ({v:.0f} days)"),
    ],
    "ndvi": [
        (0.2, "Sparse vegetation (NDVI {v:.2f})"),
        (0.5, "Moderate vegetation (NDVI {v:.2f})"),
        (float("inf"), "Dense vegetation (NDVI {v:.2f})"),
    ],
    "nbr": [
        (-0.1, "Healthy unburned vegetation (NBR {v:+.2f})"),
        (float("inf"), "Burned-vegetation signal (NBR {v:+.2f})"),
    ],
    "dnbr": [
        (0.05, "Negligible burn severity (dNBR {v:.2f})"),
        (0.3, "Low burn severity (dNBR {v:.2f})"),
        (0.66, "Moderate burn severity (dNBR {v:.2f})"),
        (float("inf"), "High burn severity (dNBR {v:.2f})"),
    ],
    "built_probability": [
        (0.2, "Low built-up surroundings (built-up {v:.0%})"),
        (0.5, "Partial built-up surroundings (built-up {v:.0%})"),
        (float("inf"), "Dense industrial/urban surroundings (built-up {v:.0%})"),
    ],
    "trees_probability": [
        (0.3, "Little forest cover (tree cover {v:.0%})"),
        (float("inf"), "Heavily forested surroundings (tree cover {v:.0%})"),
    ],
    "crops_probability": [
        (0.4, "Low cropland share (cropland {v:.0%})"),
        (float("inf"), "Dominant cropland (cropland {v:.0%})"),
    ],
    "water_probability": [
        (0.1, "Negligible water nearby (water {v:.0%})"),
        (float("inf"), "Water body nearby (water {v:.0%})"),
    ],
    "distance_to_industrial_m": [
        (1000, "Immediately beside industrial facilities ({km:.1f} km)"),
        (5000, "Within a few km of industrial facilities ({km:.1f} km)"),
        (float("inf"), "Far from industrial facilities ({km:.0f} km)"),
    ],
    "distance_to_refinery_m": [
        (500, "Immediately adjacent to a refinery ({km:.2f} km)"),
        (3000, "Close to a refinery ({km:.1f} km)"),
        (float("inf"), "Far from any known refinery ({km:.0f} km)"),
    ],
    "industrial_count_500m": [
        (1, "No industrial sites within 500 m"),
        (3, "A few industrial sites within 500 m ({v:.0f})"),
        (float("inf"), "Many industrial sites within 500 m ({v:.0f})"),
    ],
    "latitude": [(float("inf"), "Located at {v:.2f}°N")],
    "longitude": [(float("inf"), "Located at {v:.2f}°E")],
}

MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def describe(feature: str, value: float) -> str:
    """One human-readable clause for a raw feature value."""
    if feature == "burned_area_overlap":
        return ("Overlaps a previously mapped burned area"
                if value >= 0.5 else
                "No overlap with any mapped burned area")
    if feature == "day_night_flag":
        return "Night-time detection" if value >= 0.5 else "Day-time detection"
    if feature == "detection_month":
        m = int(value) if 1 <= value <= 12 else 0
        return f"Detected in {MONTHS[m]}" if m else "Seasonal timing unknown"

    bands = BANDS.get(feature)
    if bands:
        for bound, text in bands:
            if value < bound:
                return text.format(v=value, km=value / 1000.0)
    return f"{feature} = {value:.3g}"


def render_evidence(contribution: dict[str, Any],
                    predicted_class: str) -> str:
    """Render one SHAP contribution as a readable evidence string."""
    clause = describe(contribution["feature"], contribution["value"])
    stance = ("supports" if contribution["shap_value"] > 0
              else "argues against")
    return f"{clause} — {stance} {predicted_class}"


def why_not_text(predicted_class: str, second_class: str,
                 second_probability: float,
                 shap_pred: dict[str, float],
                 shap_second: dict[str, float],
                 features: dict[str, Any]) -> str:
    """Explain why the runner-up class lost, via per-feature SHAP deltas."""
    deltas = {
        f: shap_pred.get(f, 0.0) - shap_second.get(f, 0.0)
        for f in shap_pred
    }
    top = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)[:3]
    parts = []
    for feat, delta in top:
        if delta <= 0:
            continue
        value = features.get(feat)
        if value is None:
            continue
        clause = describe(feat, float(value))
        parts.append(f"{clause} points to {predicted_class} rather than "
                     f"{second_class}")
    if not parts:
        return (f"Classification as {second_class} "
                f"({second_probability:.0%}) was possible but the overall "
                f"evidence pattern fits {predicted_class} better.")
    return f"Not classified as {second_class} ({second_probability:.0%}): " \
           + "; ".join(parts) + "."


def investigation_priority(predicted_class: str,
                           probability: float) -> tuple[str, str]:
    """High / Medium / Low investigation priority.

    High: Industrial Fire or Gas Flare with calibrated probability > 70%.
    Low:  Other/Unknown, or calibrated confidence below 40%.
    Medium: everything else.
    """
    if predicted_class in ("Industrial Fire", "Gas Flare") and probability > 0.70:
        return ("High",
                f"High-consequence class ({predicted_class}) with strong "
                f"model confidence ({probability:.0%}).")
    if predicted_class == "Other/Unknown" or probability < 0.40:
        return ("Low",
                f"Weak or unconfirmed classification "
                f"({predicted_class} at {probability:.0%}).")
    return ("Medium",
            f"{predicted_class} at {probability:.0%} — plausible but not "
            f"urgent.")
