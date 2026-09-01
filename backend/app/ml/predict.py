"""Single-detection inference for the SIH26162 fire classifier.

Loads the artifacts produced by `train_model.py` and exposes
`classify_detection(feature_dict)`, which is the entrypoint the API calls.

Returns:
    - top-3 classes with calibrated probabilities
    - top-5 SHAP feature contributions for the predicted class

Run a demo:
    python -m app.ml.predict
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_VERSION = "xgboost-optuna-calibrated-v1.0"

_state: dict[str, Any] = {}


def _load_artifacts() -> dict[str, Any]:
    """Lazily load model, calibrator, explainer and metadata (cached)."""
    if _state:
        return _state
    if not os.path.isdir(ARTIFACTS_DIR):
        raise FileNotFoundError(
            f"artifacts not found at {ARTIFACTS_DIR} — "
            "run `python -m app.ml.train_model` first"
        )
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "features.json"), "r") as f:
        meta = json.load(f)

    _state["model"] = model
    _state["calibrator"] = calibrator
    _state["explainer"] = shap.TreeExplainer(model)
    _state["meta"] = meta
    return _state


def _encode_features(feature_dict: dict[str, Any]) -> list[float]:
    """Order, encode and fill features according to features.json."""
    state = _load_artifacts()
    meta = state["meta"]
    encoding = meta.get("encoding", {})
    defaults = meta.get("defaults", {})

    row: list[float] = []
    for feat in meta["features"]:
        val = feature_dict.get(feat, defaults.get(feat))
        if val is None:
            raise ValueError(f"missing feature '{feat}' and no default")
        if feat in encoding:
            val = encoding[feat].get(val, encoding[feat].get(defaults.get(feat), 0))
        try:
            row.append(float(val))
        except (TypeError, ValueError) as e:
            raise ValueError(f"feature '{feat}' has non-numeric value {val!r}") from e
    return row


def _shap_row_for_sample(shap_values, n_classes: int,
                         class_idx: int) -> np.ndarray:
    """Normalise shap output to (n_features,) for one sample / one class."""
    sv = shap_values
    if isinstance(sv, list):
        arr = np.asarray(sv[class_idx])
        return arr[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n, f)
        if sv.shape[-1] == n_classes and sv.shape[0] != n_classes:
            return sv[0, :, class_idx]
        return sv[class_idx, 0, :]
    if sv.ndim == 2:
        return sv[0]
    raise ValueError(f"unexpected shap_values shape {sv.shape}")


def classify_detection(feature_dict: dict[str, Any],
                       top_k: int = 3,
                       shap_k: int = 5,
                       with_shap: bool = True) -> dict[str, Any]:
    """Classify one thermal anomaly detection.

    Args:
        feature_dict: dict keyed by feature names (see features.json).
            `day_night_flag` accepts 'day'/'night'. Missing features fall
            back to training-set defaults.
        top_k: number of classes to return.
        shap_k: number of SHAP contributions to return.
        with_shap: skip SHAP computation (fast bulk path for lists).

    Returns:
        dict with predicted_class, confidence, top_k classes with
        calibrated probabilities, and top shap_k feature contributions
        (signed; positive pushes toward the predicted class).
    """
    state = _load_artifacts()
    meta = state["meta"]
    classes: list[str] = meta["classes"]
    calibrator = state["calibrator"]

    row = _encode_features(feature_dict)
    X = np.asarray([row], dtype=float)

    proba = calibrator.predict_proba(X)[0]
    order = np.argsort(-proba)

    top_classes = [
        {"class": classes[i], "probability": round(float(proba[i]), 4)}
        for i in order[:top_k]
    ]
    pred_idx = int(order[0])
    predicted_class = classes[pred_idx]

    contributions: list[dict[str, Any]] = []
    if with_shap:
        explainer = state["explainer"]
        sv = explainer.shap_values(X)
        sv_row = _shap_row_for_sample(sv, len(classes), pred_idx)
        shap_order = np.argsort(-np.abs(sv_row))[:shap_k]
        contributions = [
            {
                "feature": meta["features"][j],
                "value": row[j],
                "shap_value": round(float(sv_row[j]), 4),
                "direction": "increases" if sv_row[j] > 0 else "decreases",
            }
            for j in shap_order
        ]

    return {
        "predicted_class": predicted_class,
        "confidence": round(float(proba[pred_idx]), 4),
        "top_3": top_classes,
        "shap_top_5": contributions,
        "model_version": MODEL_VERSION,
    }


def explain_all_classes(feature_dict: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Signed SHAP values for every class, for one sample.

    Returns {class_name: {feature_name: shap_value}} — used by the API to
    build 'why not the runner-up class' explanations.
    """
    state = _load_artifacts()
    meta = state["meta"]
    classes: list[str] = meta["classes"]
    row = _encode_features(feature_dict)
    X = np.asarray([row], dtype=float)

    sv = state["explainer"].shap_values(X)
    out: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(classes):
        sv_row = _shap_row_for_sample(sv, len(classes), i)
        out[cls] = {
            feat: round(float(sv_row[j]), 4)
            for j, feat in enumerate(meta["features"])
        }
    return out


def _demo() -> None:
    """Demo: classify a held-out row from the training CSV."""
    import pandas as pd

    csv_path = os.path.join(BASE_DIR, "..", "data", "training_data.csv")
    df = pd.read_csv(csv_path)
    sample = df[df["class_label"] == "Gas Flare"].iloc[0].to_dict()
    true_label = sample.pop("class_label")

    result = classify_detection(sample)
    print(f"true label: {true_label}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _demo()
