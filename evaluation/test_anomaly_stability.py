import os
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD ANOMALY STABILITY EVALUATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/esif_pue_features.parquet"

SAMPLE_SIZE = 100_000

CONTAMINATIONS = [0.01, 0.02, 0.05]


print("=" * 60)
print("GREENGUARD ANOMALY STABILITY EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")


# ------------------------------------------------------------
# 2. Select modeling sample
# ------------------------------------------------------------

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

print(f"Rows used: {len(df):,}")


# ------------------------------------------------------------
# 3. Select numerical sustainability features
# ------------------------------------------------------------

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

available_features = [
    col for col in FEATURE_COLUMNS
    if col in df.columns
]

if not available_features:
    raise ValueError(
        "No expected modeling features were found."
    )

print("\nFeatures used:")

for feature in available_features:
    print(f"  - {feature}")


# ------------------------------------------------------------
# 4. Prepare ML matrix
# ------------------------------------------------------------

model_df = df[
    available_features
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

print(
    f"\nRows after cleaning: {len(model_df):,}"
)


X = model_df.values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 5. Test different contamination levels
# ------------------------------------------------------------

results = []

for contamination in CONTAMINATIONS:

    print(
        f"\nRunning Isolation Forest "
        f"(contamination={contamination:.0%})..."
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    predictions = model.fit_predict(X_scaled)

    anomaly_count = int(
        np.sum(predictions == -1)
    )

    anomaly_rate = (
        anomaly_count / len(predictions)
    )

    results.append(
        {
            "contamination": contamination,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
        }
    )


# ------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ANOMALY STABILITY RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ------------------------------------------------------------
# 7. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    """
The contamination parameter controls the expected proportion
of observations treated as anomalous.

This experiment does NOT establish ground-truth accuracy.

It tests how sensitive GreenGuard's anomaly volume is to the
chosen contamination setting.

A robust system should document this sensitivity rather than
presenting one contamination value as absolute truth.
"""
)

print("\nEvaluation completed successfully.")