"""Train the 5-class thermal anomaly classifier (SIH26162).

Pipeline:
  1. Load training_data.csv
  2. Geographic block split: India divided into ~50 km grid cells, whole
     cells assigned to train/val/test so nearby points never leak across
     splits. Cell assignment is class-aware (greedy stratification) so
     every class appears in every split despite regional biases.
  3. XGBoost multiclass with class weights (imbalanced classes)
  4. Optuna hyperparameter tuning (~30 trials) on the val split
  5. Probability calibration (Platt/sigmoid) on the val split
  6. Test-set evaluation: Macro-F1, per-class P/R/F1, confusion matrix
  7. SHAP global feature importance + summary plot
  8. Artifacts saved to backend/app/ml/artifacts/
     (model.pkl, calibrator.pkl, features.json, shap_summary.png)

Run:
    python -m app.ml.train_model
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

SEED = 26162
N_OPTUNA_TRIALS = 30
CELL_SIZE_DEG = 0.45  # ~50 km
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "..", "data", "training_data.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DAY_NIGHT_ENCODING = {"day": 0, "night": 1}
TARGET_COL = "class_label"


# ----------------------------------------------------------------------------
# Data loading & geographic block split
# ----------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Load CSV, encode day/night flag and integer-encode labels.

    Returns (df, classes, label_to_idx) where `classes` is the sorted
    label list; the encoded `y_code` column is the model target.
    """
    df = pd.read_csv(DATA_CSV)
    df["day_night_flag"] = df["day_night_flag"].map(DAY_NIGHT_ENCODING)
    if df["day_night_flag"].isna().any():
        raise ValueError("unexpected day_night_flag values in CSV")
    classes = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    df["y_code"] = df[TARGET_COL].map(label_to_idx)
    return df, classes, label_to_idx


def geographic_block_split(df: pd.DataFrame,
                           cell_size: float = CELL_SIZE_DEG,
                           seed: int = SEED) -> pd.Series:
    """Assign each row a split via whole-cell allocation on a ~50 km grid.

    Cells are allocated greedily, rarest class first, so that each split
    receives roughly its target share of every class while never splitting
    a cell across two sets (no geographic leakage).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["cell_lat"] = np.floor(df["latitude"] / cell_size)
    df["cell_lon"] = np.floor(df["longitude"] / cell_size)

    cell_class_counts: dict[tuple, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for (clat, clon, label), cnt in (
        df.groupby(["cell_lat", "cell_lon", TARGET_COL]).size().items()
    ):
        cell_class_counts[(clat, clon)][label] = cnt

    class_totals = df[TARGET_COL].value_counts().to_dict()
    # Rarest classes first so their limited cells are placed deliberately.
    class_order = sorted(class_totals, key=class_totals.get)

    cell_to_split: dict[tuple, str] = {}
    assigned_per_class: dict[str, dict[str, int]] = {
        c: {"train": 0, "val": 0, "test": 0} for c in class_order
    }

    for cls in class_order:
        cells_with_cls = [c for c, counts in cell_class_counts.items()
                          if counts.get(cls, 0) > 0]
        # Shuffle then sort by this class's count desc: big cells placed
        # first while remaining slack is largest.
        rng.shuffle(cells_with_cls)
        cells_with_cls.sort(
            key=lambda c: cell_class_counts[c].get(cls, 0), reverse=True
        )
        total = class_totals[cls]
        for cell in cells_with_cls:
            if cell in cell_to_split:
                assigned_per_class[cls][cell_to_split[cell]] += (
                    cell_class_counts[cell].get(cls, 0)
                )
                continue
            # give the cell to the split most below its target share
            deficits = {
                s: SPLIT_FRACS[s] * total - assigned_per_class[cls][s]
                for s in SPLIT_FRACS
            }
            split = max(deficits, key=deficits.get)
            cell_to_split[cell] = split
            assigned_per_class[cls][split] += cell_class_counts[cell].get(cls, 0)

    split_of_row = df.apply(
        lambda r: cell_to_split[(r["cell_lat"], r["cell_lon"])], axis=1
    )
    return split_of_row


def verify_split(df: pd.DataFrame, splits: pd.Series) -> None:
    for name in ("train", "val", "test"):
        mask = splits == name
        counts = df.loc[mask, TARGET_COL].value_counts()
        missing = set(df[TARGET_COL].unique()) - set(counts.index)
        if missing:
            raise RuntimeError(f"split '{name}' missing classes: {missing}")
        print(f"  {name:<6} n={mask.sum():>5}  cells-classes ok: "
              f"{dict(counts)}")


# ----------------------------------------------------------------------------
# Training / tuning / calibration
# ----------------------------------------------------------------------------

def make_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )


def class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    n = len(y)
    w = n / (len(counts) * counts.loc[y].to_numpy())
    return w.astype(float)


def tune(X_tr, y_tr, w_tr, X_va, y_va, n_trials: int) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = make_model(params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        return f1_score(y_va, model.predict(X_va), average="macro")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  best macro-F1 (val): {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")
    return study.best_params


# ----------------------------------------------------------------------------
# SHAP helpers
# ----------------------------------------------------------------------------

def shap_per_class(explainer: shap.TreeExplainer, X: pd.DataFrame,
                   n_classes: int) -> np.ndarray:
    """Return SHAP values shaped (n_classes, n_samples, n_features)."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.stack([np.asarray(a) for a in sv], axis=0)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] == n_classes and sv.shape[0] != n_classes:
            return np.moveaxis(sv, -1, 0)
        return sv
    if sv.ndim == 2:  # binary-style single output
        return sv[None, :, :]
    raise ValueError(f"unexpected shap_values shape {sv.shape}")


def save_shap_summary(model, X_test: pd.DataFrame, classes: list[str],
                      out_png: str, out_csv: str) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    sv = shap_per_class(explainer, X_test, len(classes))  # (C, N, F)
    mean_abs = np.abs(sv).mean(axis=(0, 1))               # (F,)

    imp = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap_global": mean_abs,
    }).sort_values("mean_abs_shap_global", ascending=False)
    for i, cls in enumerate(classes):
        imp[f"mean_abs_shap__{cls}"] = (
            np.abs(sv[i]).mean(axis=0)[X_test.columns.get_indexer(imp["feature"])]
        )
    imp = imp.reset_index(drop=True)
    imp.to_csv(out_csv, index=False)

    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap_global"], color="#ff6b35")
    ax.set_xlabel("mean(|SHAP value|) — global, all classes")
    ax.set_title("Feature importance (SHAP) — SIH26162 fire classifier")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return imp


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("[1/7] loading data ...")
    df, classes, label_to_idx = load_data()
    feature_cols = [c for c in df.columns
                    if c not in (TARGET_COL, "y_code")]

    print("[2/7] geographic 50km block split ...")
    splits = geographic_block_split(df)
    verify_split(df, splits)

    tr, va, te = (splits == "train"), (splits == "val"), (splits == "test")
    X_tr, y_tr = df.loc[tr, feature_cols], df.loc[tr, "y_code"]
    X_va, y_va = df.loc[va, feature_cols], df.loc[va, "y_code"]
    X_te, y_te = df.loc[te, feature_cols], df.loc[te, "y_code"]
    w_tr = class_weights(y_tr)

    print("[3/7] optuna tuning "
          f"({N_OPTUNA_TRIALS} trials, train->val) ...")
    best_params = tune(X_tr, y_tr, w_tr, X_va, y_va, N_OPTUNA_TRIALS)

    print("[4/7] fitting best model on train ...")
    model = make_model(best_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    print("[5/7] calibrating probabilities on val (sigmoid) ...")
    # FrozenEstimator = 'prefit' pattern (removed in newer sklearn):
    # the XGB model is already trained; only the Platt-scaling calibrator
    # is fit here, on the validation split.
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(model), method="sigmoid"
    )
    calibrator.fit(X_va, y_va)

    print("[6/7] evaluating on held-out test cells ...")
    proba = calibrator.predict_proba(X_te)
    preds = calibrator.predict(X_te)
    macro_f1 = f1_score(y_te, preds, average="macro")
    print(f"\n  Macro-F1 (test): {macro_f1:.4f}\n")
    print("  Per-class precision / recall / F1:")
    report = classification_report(
        y_te, preds, labels=range(len(classes)),
        target_names=classes, digits=3, zero_division=0,
    )
    print("\n".join("    " + line for line in report.splitlines()))
    cm = confusion_matrix(y_te, preds, labels=range(len(classes)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{c}" for c in classes],
        columns=[f"pred:{c}" for c in classes],
    )
    print("\n  Confusion matrix:")
    print("\n".join("    " + line for line in cm_df.to_string().splitlines()))

    print("\n[7/7] SHAP global importance ...")
    imp = save_shap_summary(
        model, X_te, classes,
        os.path.join(ARTIFACTS_DIR, "shap_summary.png"),
        os.path.join(ARTIFACTS_DIR, "shap_importance.csv"),
    )
    print("\n  Top 10 features (global mean |SHAP|):")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:<26} {r['mean_abs_shap_global']:.4f}")

    # ---- save artifacts ----
    defaults = {
        c: (float(df[c].median()) if c != "day_night_flag"
            else int(df[c].mode().iloc[0]))
        for c in feature_cols
    }
    features_meta = {
        "features": feature_cols,
        "classes": classes,
        "encoding": {"day_night_flag": DAY_NIGHT_ENCODING},
        "defaults": defaults,
        "best_params": best_params,
        "cell_size_deg": CELL_SIZE_DEG,
        "metrics": {
            "test_macro_f1": round(float(macro_f1), 4),
            "test_samples": int(len(y_te)),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(ARTIFACTS_DIR, "features.json"), "w") as f:
        json.dump(features_meta, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACTS_DIR}:")
    print("  model.pkl, calibrator.pkl, features.json,")
    print("  shap_summary.png, shap_importance.csv")


if __name__ == "__main__":
    main()
