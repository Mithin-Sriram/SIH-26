"""Generate a realistic synthetic training dataset for the 5-class
thermal-anomaly classifier (SIH26162).

Each sample is a detected thermal anomaly described by thermal, historical,
spectral (Sentinel-2 proxy), land-cover (Dynamic World proxy), geospatial
(OSM proxy) and burn features. Class-conditional distributions are designed
so that each class has a learnable signature, with 10-20% inter-class
overlap injected via feature blending + Gaussian noise so the problem is
not trivially separable.

Run:
    python -m app.data.generate_synthetic_data
or:
    python backend/app/data/generate_synthetic_data.py

Output:
    backend/app/data/training_data.csv  (3,300 rows + header)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RNG_SEED = 26162
INDIA_LAT = (6.5, 35.5)
INDIA_LON = (68.0, 97.5)

CLASS_COUNTS: dict[str, int] = {
    "Gas Flare": 600,
    "Industrial Fire": 500,
    "Wildfire": 1000,
    "Agricultural Burning": 800,
    "Other/Unknown": 400,
}
assert sum(CLASS_COUNTS.values()) == 3300

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "frp_mw",
    "brightness_temp_i4_k",
    "brightness_temp_i5_k",
    "confidence_pct",
    "day_night_flag",
    "persistence_days",
    "total_detections_365d",
    "frp_std_dev_pct",
    "day_night_ratio",
    "active_days_count",
    "ndvi",
    "nbr",
    "dnbr",
    "built_probability",
    "trees_probability",
    "crops_probability",
    "water_probability",
    "distance_to_industrial_m",
    "distance_to_refinery_m",
    "industrial_count_500m",
    "burned_area_overlap",
    "detection_month",
    "class_label",
]


def _trunc(mean: float, std: float, n: int, low: float, high: float,
           rng: np.random.Generator) -> np.ndarray:
    """Truncated-normal sample clipped to [low, high]."""
    return np.clip(rng.normal(mean, std, n), low, high)


def _trunc_nonneg(mean: float, std: float, n: int, rng: np.random.Generator,
                  high: float = 1e9) -> np.ndarray:
    return np.clip(rng.normal(mean, std, n), 0.0, high)


def _norm_probs(p: np.ndarray) -> np.ndarray:
    """Row-normalize a (n,4) probability matrix so rows sum to 1."""
    p = np.clip(p, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


@dataclass
class ClassProfile:
    name: str
    n: int
    lat_centre: tuple[float, float]      # (mean, std)
    lon_centre: tuple[float, float]
    frp: tuple[float, float]
    bt_i4: tuple[float, float]
    bt_i5: tuple[float, float]
    conf: tuple[float, float]
    day_night_p_night: float            # prob of 'night'
    persist: tuple[float, float]
    tot_365: tuple[float, float]
    frp_std: tuple[float, float]        # percent (0-100)
    dnr: tuple[float, float]            # day_night_ratio mean/std
    active_days: tuple[float, float]
    ndvi: tuple[float, float]
    nbr: tuple[float, float]
    dnbr: tuple[float, float]
    built: tuple[float, float]
    trees: tuple[float, float]
    crops: tuple[float, float]
    water: tuple[float, float]
    dist_industrial: tuple[float, float]
    dist_refinery: tuple[float, float]
    ind_count_500m: tuple[float, float]
    burned_p_true: float               # prob burned_area_overlap = True
    month_bias: tuple[float, float] | None  # (mean, std) month, None=uniform


PROFILES: dict[str, ClassProfile] = {
    "Gas Flare": ClassProfile(
        "Gas Flare", 600,
        lat_centre=(24.0, 3.0), lon_centre=(71.0, 2.5),
        frp=(220.0, 70.0), bt_i4=(1850.0, 180.0), bt_i5=(335.0, 12.0),
        conf=(92.0, 4.0), day_night_p_night=0.78,
        persist=(240.0, 70.0), tot_365=(290.0, 50.0),
        frp_std=(9.0, 2.5), dnr=(0.74, 0.06),
        active_days=(240.0, 70.0),
        ndvi=(0.12, 0.06), nbr=(-0.05, 0.05), dnbr=(0.05, 0.04),
        built=(0.35, 0.10), trees=(0.06, 0.04), crops=(0.10, 0.05),
        water=(0.02, 0.02),
        dist_industrial=(1400.0, 700.0), dist_refinery=(180.0, 120.0),
        ind_count_500m=(0.6, 0.7), burned_p_true=0.02,
        month_bias=None,
    ),
    "Industrial Fire": ClassProfile(
        "Industrial Fire", 500,
        lat_centre=(22.0, 3.2), lon_centre=(78.5, 5.0),
        frp=(360.0, 120.0), bt_i4=(620.0, 130.0), bt_i5=(355.0, 14.0),
        conf=(84.0, 6.0), day_night_p_night=0.5,
        persist=(28.0, 18.0), tot_365=(70.0, 40.0),
        frp_std=(32.0, 8.0), dnr=(0.48, 0.08),
        active_days=(22.0, 16.0),
        ndvi=(0.18, 0.08), nbr=(-0.02, 0.06), dnbr=(0.12, 0.06),
        built=(0.62, 0.10), trees=(0.10, 0.05), crops=(0.08, 0.04),
        water=(0.04, 0.03),
        dist_industrial=(350.0, 220.0), dist_refinery=(6000.0, 3500.0),
        ind_count_500m=(4.5, 2.0), burned_p_true=0.08,
        month_bias=None,
    ),
    "Wildfire": ClassProfile(
        "Wildfire", 1000,
        lat_centre=(27.5, 3.5), lon_centre=(84.0, 5.0),
        frp=(180.0, 90.0), bt_i4=(480.0, 110.0), bt_i5=(345.0, 16.0),
        conf=(78.0, 9.0), day_night_p_night=0.42,
        persist=(8.0, 6.0), tot_365=(12.0, 8.0),
        frp_std=(48.0, 14.0), dnr=(0.45, 0.09),
        active_days=(7.0, 5.5),
        ndvi=(0.45, 0.14), nbr=(-0.35, 0.16), dnbr=(0.78, 0.16),
        built=(0.07, 0.04), trees=(0.66, 0.12), crops=(0.06, 0.04),
        water=(0.03, 0.03),
        dist_industrial=(18000.0, 9000.0), dist_refinery=(32000.0, 16000.0),
        ind_count_500m=(0.05, 0.2), burned_p_true=0.92,
        month_bias=(4.0, 1.5),   # pre-monsoon fire season Mar-May
    ),
    "Agricultural Burning": ClassProfile(
        "Agricultural Burning", 800,
        lat_centre=(28.5, 2.2), lon_centre=(76.5, 2.5),
        frp=(55.0, 25.0), bt_i4=(380.0, 45.0), bt_i5=(332.0, 10.0),
        conf=(76.0, 8.0), day_night_p_night=0.28,
        persist=(4.0, 3.0), tot_365=(18.0, 9.0),
        frp_std=(34.0, 10.0), dnr=(0.35, 0.08),
        active_days=(4.0, 3.0),
        ndvi=(0.42, 0.12), nbr=(0.06, 0.08), dnbr=(0.18, 0.06),
        built=(0.10, 0.05), trees=(0.08, 0.04), crops=(0.66, 0.12),
        water=(0.04, 0.03),
        dist_industrial=(9000.0, 5000.0), dist_refinery=(18000.0, 9000.0),
        ind_count_500m=(0.15, 0.4), burned_p_true=0.15,
        month_bias=(10.7, 0.6),   # Oct-Nov stubble season
    ),
    "Other/Unknown": ClassProfile(
        "Other/Unknown", 400,
        lat_centre=(20.0, 7.0), lon_centre=(82.0, 8.0),
        frp=(18.0, 14.0), bt_i4=(345.0, 35.0), bt_i5=(315.0, 14.0),
        conf=(32.0, 12.0), day_night_p_night=0.5,
        persist=(6.0, 6.0), tot_365=(8.0, 6.0),
        frp_std=(40.0, 20.0), dnr=(0.5, 0.15),
        active_days=(6.0, 6.0),
        ndvi=(0.35, 0.22), nbr=(0.02, 0.18), dnbr=(0.06, 0.12),
        built=(0.25, 0.15), trees=(0.25, 0.15), crops=(0.25, 0.15),
        water=(0.10, 0.08),
        dist_industrial=(6000.0, 7000.0), dist_refinery=(14000.0, 12000.0),
        ind_count_500m=(0.4, 0.8), burned_p_true=0.05,
        month_bias=None,
    ),
}


def _sample_profile(p: ClassProfile, rng: np.random.Generator) -> pd.DataFrame:
    n = p.n

    lat = np.clip(rng.normal(*p.lat_centre, n), INDIA_LAT[0], INDIA_LAT[1])
    lon = np.clip(rng.normal(*p.lon_centre, n), INDIA_LON[0], INDIA_LON[1])
    frp = _trunc_nonneg(*p.frp, n, rng)
    bt_i4 = _trunc(*p.bt_i4, n, 200.0, 2400.0, rng)
    bt_i5 = _trunc(*p.bt_i5, n, 290.0, 400.0, rng)
    conf = _trunc(*p.conf, n, 0.0, 100.0, rng)
    night = rng.random(n) < p.day_night_p_night
    day_night_flag = np.where(night, "night", "day")

    persist = _trunc_nonneg(*p.persist, n, rng)
    tot_365 = _trunc_nonneg(*p.tot_365, n, rng).astype(int)
    frp_std = _trunc(*p.frp_std, n, 1.0, 100.0, rng)
    dnr = _trunc(*p.dnr, n, 0.0, 1.0, rng)
    active_days = _trunc_nonneg(*p.active_days, n, rng).astype(int)

    ndvi = _trunc(*p.ndvi, n, -0.2, 1.0, rng)
    nbr = _trunc(*p.nbr, n, -1.0, 1.0, rng)
    dnbr = _trunc(*p.dnbr, n, -0.3, 2.0, rng)

    probs = _norm_probs(np.column_stack([
        _trunc(*p.built, n, 0.0, 1.0, rng),
        _trunc(*p.trees, n, 0.0, 1.0, rng),
        _trunc(*p.crops, n, 0.0, 1.0, rng),
        _trunc(*p.water, n, 0.0, 1.0, rng),
    ]))
    built, trees, crops, water = probs.T

    dist_ind = _trunc_nonneg(*p.dist_industrial, n, rng)
    dist_ref = _trunc_nonneg(*p.dist_refinery, n, rng)
    ind_count = np.clip(rng.normal(*p.ind_count_500m, n), 0, None).astype(int)
    burned = (rng.random(n) < p.burned_p_true).astype(int)

    if p.month_bias is None:
        month = rng.integers(1, 13, n)
    else:
        month = np.clip(rng.normal(*p.month_bias, n), 1, 12).astype(int)

    df = pd.DataFrame({
        "latitude": np.round(lat, 5),
        "longitude": np.round(lon, 5),
        "frp_mw": np.round(frp, 2),
        "brightness_temp_i4_k": np.round(bt_i4, 1),
        "brightness_temp_i5_k": np.round(bt_i5, 1),
        "confidence_pct": np.round(conf, 1),
        "day_night_flag": day_night_flag,
        "persistence_days": np.round(persist, 1),
        "total_detections_365d": tot_365,
        "frp_std_dev_pct": np.round(frp_std, 1),
        "day_night_ratio": np.round(dnr, 3),
        "active_days_count": active_days,
        "ndvi": np.round(ndvi, 3),
        "nbr": np.round(nbr, 3),
        "dnbr": np.round(dnbr, 3),
        "built_probability": np.round(built, 3),
        "trees_probability": np.round(trees, 3),
        "crops_probability": np.round(crops, 3),
        "water_probability": np.round(water, 3),
        "distance_to_industrial_m": np.round(dist_ind, 0).astype(int),
        "distance_to_refinery_m": np.round(dist_ref, 0).astype(int),
        "industrial_count_500m": ind_count,
        "burned_area_overlap": burned,
        "detection_month": month,
        "class_label": p.name,
    })
    return df


# Semantically adjacent classes used for overlap blending: samples from
# one class are partially mixed with a *confusable* neighbour, which is
# more realistic than blending with an arbitrary class.
ADJACENT_CLASSES: dict[str, list[str]] = {
    "Gas Flare": ["Industrial Fire", "Other/Unknown"],
    "Industrial Fire": ["Gas Flare", "Other/Unknown"],
    "Wildfire": ["Agricultural Burning", "Other/Unknown"],
    "Agricultural Burning": ["Wildfire", "Other/Unknown"],
    "Other/Unknown": ["Industrial Fire", "Wildfire",
                      "Agricultural Burning", "Gas Flare"],
}


def _inject_overlap(classes: dict[str, pd.DataFrame],
                    rng: np.random.Generator,
                    blend_frac: float = 0.18) -> dict[str, pd.DataFrame]:
    """Blend a fraction of each class's features with a random sample from
    an adjacent (confusable) class to create 10-20% inter-class overlap.

    Blended rows are tagged in a temporary `_blended` column so the
    signature-enforcement pass can leave them alone. Continuous features
    are blended by weighted average; `burned_area_overlap` is taken from
    the donor half the time (otherwise it acts as a perfect giveaway).
    `day_night_flag` and `class_label` always stay with the true class.
    """
    numeric_cols = [
        "frp_mw", "brightness_temp_i4_k", "brightness_temp_i5_k",
        "confidence_pct", "persistence_days", "total_detections_365d",
        "frp_std_dev_pct", "day_night_ratio", "active_days_count",
        "ndvi", "nbr", "dnbr", "built_probability", "trees_probability",
        "crops_probability", "water_probability", "distance_to_industrial_m",
        "distance_to_refinery_m", "industrial_count_500m", "detection_month",
    ]
    int_cols = {
        "total_detections_365d", "active_days_count", "industrial_count_500m",
        "distance_to_industrial_m", "distance_to_refinery_m", "detection_month",
    }
    out = {}
    for label, df in classes.items():
        df = df.copy().reset_index(drop=True)
        df["_blended"] = 0
        n = len(df)
        k = max(1, int(round(blend_frac * n)))
        idx = rng.choice(n, size=k, replace=False)
        for i in idx:
            other = rng.choice(ADJACENT_CLASSES[label])
            donor = classes[other]
            j = rng.integers(0, len(donor))
            w = rng.uniform(0.35, 0.6)
            for col in numeric_cols:
                a = float(df.at[i, col])
                b = float(donor.at[donor.index[j], col])
                mixed = (1.0 - w) * a + w * b
                if col in int_cols:
                    df.at[i, col] = int(round(mixed))
                else:
                    df.at[i, col] = round(mixed, 3)
            if rng.random() < 0.5:
                df.at[i, "burned_area_overlap"] = int(
                    donor.at[donor.index[j], "burned_area_overlap"]
                )
            df.at[i, "_blended"] = 1
        out[label] = df
    return out


def _add_gaussian_noise(classes: dict[str, pd.DataFrame],
                        rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Add small independent noise on top of blended features."""
    out = {}
    for label, df in classes.items():
        df = df.copy()
        n = len(df)
        df["frp_mw"] += rng.normal(0, 6.0, n)
        df["brightness_temp_i4_k"] += rng.normal(0, 15.0, n)
        df["brightness_temp_i5_k"] += rng.normal(0, 3.0, n)
        df["confidence_pct"] += rng.normal(0, 2.5, n)
        df["persistence_days"] += rng.normal(0, 0.6, n)
        df["frp_std_dev_pct"] += rng.normal(0, 1.5, n)
        df["day_night_ratio"] += rng.normal(0, 0.02, n)
        df["ndvi"] += rng.normal(0, 0.015, n)
        df["nbr"] += rng.normal(0, 0.02, n)
        df["dnbr"] += rng.normal(0, 0.03, n)
        for c in ["built_probability", "trees_probability",
                  "crops_probability", "water_probability"]:
            df[c] += rng.normal(0, 0.01, n)
        out[label] = df
    return out


def _enforce_signatures(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Re-clip class-defining features into their signature range for the
    *non-blended* majority, so the core signal survives the noise pass.
    Blended rows are exempt (they are the intended overlap)."""
    df = df.copy()
    blended = df["_blended"] == 1
    gf = (df["class_label"] == "Gas Flare") & ~blended
    wf = (df["class_label"] == "Wildfire") & ~blended
    ab = (df["class_label"] == "Agricultural Burning") & ~blended

    # Gas flares stay very hot (i4 > 1600) for ~85% of samples.
    mask = gf & (rng.random(len(df)) < 0.85) & (df["brightness_temp_i4_k"] < 1600)
    n = int(mask.sum())
    if n:
        df.loc[mask, "brightness_temp_i4_k"] = rng.normal(1850, 150, n)

    # Wildfires keep dnbr > 0.5 for ~85%.
    mask = wf & (rng.random(len(df)) < 0.85) & (df["dnbr"] < 0.5)
    n = int(mask.sum())
    if n:
        df.loc[mask, "dnbr"] = rng.normal(0.78, 0.14, n)

    # Agri burning keeps dnbr < 0.3 for ~85%.
    mask = ab & (rng.random(len(df)) < 0.85) & (df["dnbr"] > 0.3)
    n = int(mask.sum())
    if n:
        df.loc[mask, "dnbr"] = rng.normal(0.18, 0.05, n)

    df["brightness_temp_i4_k"] = df["brightness_temp_i4_k"].clip(200, 2400)
    df["brightness_temp_i5_k"] = df["brightness_temp_i5_k"].clip(290, 400)
    df["confidence_pct"] = df["confidence_pct"].clip(0, 100)
    df["frp_mw"] = df["frp_mw"].clip(0, None)
    df["persistence_days"] = df["persistence_days"].clip(0, None)
    df["frp_std_dev_pct"] = df["frp_std_dev_pct"].clip(1, 100)
    df["day_night_ratio"] = df["day_night_ratio"].clip(0, 1)
    df["ndvi"] = df["ndvi"].clip(-0.2, 1.0)
    df["nbr"] = df["nbr"].clip(-1.0, 1.0)
    df["dnbr"] = df["dnbr"].clip(-0.3, 2.0)
    for c in ["built_probability", "trees_probability",
              "crops_probability", "water_probability"]:
        df[c] = df[c].clip(0, 1)
    df["total_detections_365d"] = df["total_detections_365d"].clip(0).astype(int)
    df["active_days_count"] = df["active_days_count"].clip(0).astype(int)
    df["industrial_count_500m"] = df["industrial_count_500m"].clip(0).astype(int)
    df["detection_month"] = df["detection_month"].clip(1, 12).astype(int)
    return df


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    classes = {label: _sample_profile(p, rng) for label, p in PROFILES.items()}
    classes = _inject_overlap(classes, rng, blend_frac=0.18)
    classes = _add_gaussian_noise(classes, rng)
    df = pd.concat(classes.values(), ignore_index=True)
    df = _enforce_signatures(df, rng)
    df = df.drop(columns=["_blended"])
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return df[FEATURE_COLUMNS]


def _print_summary(df: pd.DataFrame) -> None:
    print("=" * 72)
    print(f"Generated {len(df)} samples -> training_data.csv")
    print("-" * 72)
    print("Class counts:")
    for label, count in df["class_label"].value_counts().items():
        print(f"  {label:<22} {count:>5}")
    print("-" * 72)
    print("Feature ranges (numeric):")
    numeric = [c for c in df.columns if c not in
               {"class_label", "day_night_flag", "burned_area_overlap"}]
    for col in numeric:
        lo, hi = df[col].min(), df[col].max()
        mu = df[col].mean()
        print(f"  {col:<26} min={lo:>12.3f}  max={hi:>12.3f}  mean={mu:>12.3f}")
    print("-" * 72)
    print("Categorical breakdown:")
    print(df["day_night_flag"].value_counts().to_string().replace("\n", "\n  "))
    print("  burned_area_overlap=1:", int(df["burned_area_overlap"].sum()))
    print("=" * 72)


def main() -> None:
    df = generate()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "training_data.csv")
    df.to_csv(out_path, index=False)
    _print_summary(df)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
