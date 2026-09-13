import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "plug_and_light_kw",
    "pue",
    "cooling_to_it_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "non_it_to_it_ratio",
]


def prepare_ml_data(df):
    """
    Prepare sustainability features for anomaly detection.
    """

    ml_df = df.copy()

    # Keep only features required by the model
    ml_df = ml_df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    # Remove infinite values
    ml_df = ml_df[
        ml_df[FEATURE_COLUMNS]
        .replace([float("inf"), float("-inf")], pd.NA)
        .notna()
        .all(axis=1)
    ].copy()

    return ml_df


def train_anomaly_detector(
    df,
    contamination=0.02,
    random_state=42
):
    """
    Train Isolation Forest for data-center
    operational anomaly detection.
    """

    ml_df = prepare_ml_data(df)

    X = ml_df[FEATURE_COLUMNS]

    # Standardize features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_scaled)

    # Predictions
    predictions = model.predict(X_scaled)

    # Isolation Forest:
    # 1 = normal
    # -1 = anomaly

    ml_df["anomaly_label"] = predictions

    # Convert decision function into an intuitive score.
    # Larger values = more anomalous.
    ml_df["anomaly_score"] = (
        -model.decision_function(X_scaled)
    )

    return model, scaler, ml_df