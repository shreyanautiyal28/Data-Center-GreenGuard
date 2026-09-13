import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS_V2 = [
    "it_power_kw",
    "pue",
    "cooling_to_it_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "non_it_to_it_ratio",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
]


def prepare_v2_features(df):
    """
    Prepare GreenGuard V2 sustainability features.

    Expected-behavior deviations are included only for
    HVAC and pump because those passed the baseline
    quality gate.

    Cooling deviation is intentionally excluded.
    """

    df = df.copy()

    # ======================================================
    # RECONSTRUCT SUSTAINABILITY RATIOS
    # ======================================================

    required_raw = [
        "it_power_kw",
        "cooling_kw",
        "hvac_kw",
        "pump_kw",
        "pue",
        "expected_hvac_kw",
        "expected_pump_kw",
    ]

    missing = [
        col for col in required_raw
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required V2 raw columns: "
            + ", ".join(missing)
        )

    safe_it = df["it_power_kw"].clip(
        lower=1
    )

    df["cooling_to_it_ratio"] = (
        df["cooling_kw"] / safe_it
    )

    df["hvac_to_it_ratio"] = (
        df["hvac_kw"] / safe_it
    )

    df["pump_to_it_ratio"] = (
        df["pump_kw"] / safe_it
    )

    non_it_power = (
        df["cooling_kw"]
        + df["hvac_kw"]
        + df["pump_kw"]
    )

    df["non_it_to_it_ratio"] = (
        non_it_power / safe_it
    )

    # ======================================================
    # EXPECTED-BEHAVIOR DEVIATIONS
    # ======================================================

    df["hvac_deviation_ratio"] = (
        (
            df["hvac_kw"]
            - df["expected_hvac_kw"]
        )
        / df["expected_hvac_kw"].clip(
            lower=1.0
        )
    )

    df["pump_deviation_ratio"] = (
        (
            df["pump_kw"]
            - df["expected_pump_kw"]
        )
        / df["expected_pump_kw"].clip(
            lower=1.0
        )
    )

    # ======================================================
    # FINAL FEATURES
    # ======================================================

    features = df[
        FEATURE_COLUMNS_V2
    ].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = (
        features
        .notna()
        .all(axis=1)
    )

    features = features.loc[
        valid_mask
    ]

    return features, valid_mask


def train_anomaly_detector_v2(
    df,
    contamination=0.02,
    random_state=42,
):
    """
    Train GreenGuard Anomaly Detector V2.
    """

    features, valid_mask = (
        prepare_v2_features(df)
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        features
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X)

    scores = model.decision_function(
        X
    )

    labels = model.predict(
        X
    )

    results = df.loc[
        valid_mask
    ].copy()

    # Preserve reconstructed features
    for column in FEATURE_COLUMNS_V2:
        results[column] = (
            features[column]
        )

    results["anomaly_label_v2"] = labels

    results["anomaly_score_v2"] = (
        -scores
    )

    return (
        model,
        scaler,
        results,
    )