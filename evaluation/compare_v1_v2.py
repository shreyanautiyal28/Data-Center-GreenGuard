import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_PATH = (
    "data/processed/"
    "rolling_baseline_validation.parquet"
)


print("=" * 70)
print("GREENGUARD V1 vs V2 ANOMALY DETECTOR COMPARISON")
print("=" * 70)


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_parquet(INPUT_PATH)

print(
    f"\nRows loaded: {len(df):,}"
)


# ==========================================================
# RECONSTRUCT V1 FEATURES
# ==========================================================

safe_it = df["it_power_kw"].clip(lower=1)

df["cooling_to_it_ratio"] = (
    df["cooling_kw"] / safe_it
)

df["hvac_to_it_ratio"] = (
    df["hvac_kw"] / safe_it
)

df["pump_to_it_ratio"] = (
    df["pump_kw"] / safe_it
)

non_it = (
    df["cooling_kw"]
    + df["hvac_kw"]
    + df["pump_kw"]
)

df["non_it_to_it_ratio"] = (
    non_it / safe_it
)


# ==========================================================
# V1 FEATURES
# ==========================================================

V1_FEATURES = [
    "it_power_kw",
    "pue",
    "cooling_to_it_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "non_it_to_it_ratio",
]


# ==========================================================
# V2 FEATURES
# ==========================================================

V2_FEATURES = [
    "it_power_kw",
    "pue",
    "cooling_to_it_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "non_it_to_it_ratio",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
]


# ==========================================================
# COMMON VALID DATA
# ==========================================================

all_features = (
    V2_FEATURES
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

valid = df[
    all_features
].notna().all(axis=1)

df = df.loc[
    valid
].copy()

print(
    f"Common modeling rows: "
    f"{len(df):,}"
)


# ==========================================================
# TRAIN V1
# ==========================================================

print(
    "\nTraining V1..."
)

scaler_v1 = StandardScaler()

X_v1 = scaler_v1.fit_transform(
    df[V1_FEATURES]
)

model_v1 = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model_v1.fit(X_v1)

score_v1 = (
    -model_v1.decision_function(X_v1)
)

label_v1 = model_v1.predict(X_v1)


# ==========================================================
# TRAIN V2
# ==========================================================

print(
    "Training V2..."
)

scaler_v2 = StandardScaler()

X_v2 = scaler_v2.fit_transform(
    df[V2_FEATURES]
)

model_v2 = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model_v2.fit(X_v2)

score_v2 = (
    -model_v2.decision_function(X_v2)
)

label_v2 = model_v2.predict(X_v2)


# ==========================================================
# STORE
# ==========================================================

df["v1_anomaly"] = (
    label_v1 == -1
)

df["v2_anomaly"] = (
    label_v2 == -1
)

df["v1_score"] = score_v1

df["v2_score"] = score_v2


# ==========================================================
# BASIC COMPARISON
# ==========================================================

v1_count = df["v1_anomaly"].sum()

v2_count = df["v2_anomaly"].sum()

overlap = (
    df["v1_anomaly"]
    & df["v2_anomaly"]
).sum()

v1_only = (
    df["v1_anomaly"]
    & ~df["v2_anomaly"]
).sum()

v2_only = (
    df["v2_anomaly"]
    & ~df["v1_anomaly"]
).sum()


print(
    "\n" + "=" * 70
)

print(
    "V1 vs V2 RESULTS"
)

print(
    "=" * 70
)

print(
    f"V1 anomalies       : {v1_count:,}"
)

print(
    f"V2 anomalies       : {v2_count:,}"
)

print(
    f"Both models        : {overlap:,}"
)

print(
    f"V1 only            : {v1_only:,}"
)

print(
    f"V2 only            : {v2_only:,}"
)

print(
    f"V1/V2 overlap rate : "
    f"{overlap / max(v1_count, 1) * 100:.2f}%"
)


# ==========================================================
# V2 ANOMALY CHARACTERISTICS
# ==========================================================

v2 = df[
    df["v2_anomaly"]
].copy()

print(
    "\n" + "=" * 70
)

print(
    "V2 ANOMALY CHARACTERISTICS"
)

print(
    "=" * 70
)

print(
    f"\nMedian PUE: "
    f"{v2['pue'].median():.4f}"
)

print(
    f"Median HVAC deviation: "
    f"{v2['hvac_deviation_ratio'].median():.4f}"
)

print(
    f"Median Pump deviation: "
    f"{v2['pump_deviation_ratio'].median():.4f}"
)

print(
    f"> +100% HVAC deviation: "
    f"{(v2['hvac_deviation_ratio'] > 1).mean() * 100:.2f}%"
)

print(
    f"> +100% Pump deviation: "
    f"{(v2['pump_deviation_ratio'] > 1).mean() * 100:.2f}%"
)


# ==========================================================
# TOP V2 ANOMALIES
# ==========================================================

print(
    "\nTop 10 V2 anomalies:"
)

columns = [
    "ts",
    "it_power_kw",
    "pue",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "v1_score",
    "v2_score",
]

print(
    v2
    .sort_values(
        "v2_score",
        ascending=False
    )
    .head(10)[columns]
    .to_string(index=False)
)


# ==========================================================
# SAVE
# ==========================================================

output = (
    "data/processed/"
    "v1_v2_comparison.parquet"
)

df.to_parquet(
    output,
    index=False
)

print(
    f"\nSaved:\n{output}"
)

print(
    "\nV1 vs V2 comparison completed."
)