"""Classification model loading and prediction logic.

This is a prototype classifier. For the hackathon MVP it uses a simple
rule-based heuristic over the feature vector so the pipeline runs without
a trained artifact. The interface mirrors a real sklearn model so we can
swap in a trained `joblib` model later (see `load_model`).
"""

from __future__ import annotations

from ..models.schemas import (
    ClassificationResponse,
    DetectionFeatures,
    FireCategory,
)

MODEL_VERSION = "rule-based-prototype-v0.1"


def load_model():
    """Return a placeholder model handle.

    In production this would `joblib.load("model.joblib")`. For the
    prototype we return None and fall back to rule-based classification.
    """
    return None


def _rule_based_predict(features: DetectionFeatures) -> tuple[FireCategory, dict[str, float]]:
    """Heuristic classifier over the feature vector.

    Returns the predicted category plus a (unnormalized) score dict that
    we softmax-normalize into pseudo-probabilities for display.
    """

    scores: dict[str, float] = {c.value: 0.1 for c in FireCategory}

    if features.distance_to_gas_infra_km < 5.0 and features.frp_mw > 300:
        scores[FireCategory.GAS_FLARE.value] += 2.0
    if features.distance_to_industry_km < 5.0 and features.frp_mw > 200:
        scores[FireCategory.INDUSTRIAL_FIRE.value] += 2.0
    if features.ndvi > 0.5 and features.area_km2 > 1.0:
        scores[FireCategory.WILDFIRE.value] += 1.5
    if features.distance_to_farmland_km < 2.0 and features.persistence_hours < 6:
        scores[FireCategory.AGRICULTURAL_BURNING.value] += 1.5

    if max(scores.values()) < 0.6:
        scores[FireCategory.OTHER_UNKNOWN.value] += 1.0

    # softmax normalize
    import math

    max_score = max(scores.values())
    exps = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exps.values())
    probabilities = {k: round(v / total, 4) for k, v in exps.items()}

    category = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    return FireCategory(category), probabilities


def classify(features: DetectionFeatures) -> ClassificationResponse:
    model = load_model()
    if model is not None:
        # Real model path (future): feature array -> model.predict_proba
        raise NotImplementedError("Trained model inference not wired yet.")

    category, probabilities = _rule_based_predict(features)
    return ClassificationResponse(
        category=category,
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )
