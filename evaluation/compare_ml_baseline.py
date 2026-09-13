import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD ML vs SIMPLE BASELINE EVALUATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/esif_pue_features.parquet"

SAMPLE_SIZE = 100_000

# PUE thresholds used for simple operational baselines
PUE_THRESHOLDS = [1.10, 1.20, 1.30, 1.50]


print("=" * 60)
print("GREENGUARD ML VS SIMPLE BASELINE")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")


# ------------------------------------------------------------
# 2. Use same sample size as initial ML experiment
# ------------------------------------------------------------

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

print(f"Rows selected: {len(df):,}")


# ------------------------------------------------------------
# 3. Required features
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

if "pue" not in df.columns:
    raise ValueError("PUE column is required.")

model_df = df[
    available_features
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

print(f"Rows after cleaning: {len(model_df):,}")


# ------------------------------------------------------------
# 4. Isolation Forest
# ------------------------------------------------------------

print("\nRunning Isolation Forest...")

X = model_df[
    available_features
].values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

ml_predictions = model.fit_predict(X_scaled)

ml_anomalies = (
    ml_predictions == -1
)

ml_count = int(
    ml_anomalies.sum()
)

ml_rate = (
    ml_count / len(model_df)
)


# ------------------------------------------------------------
# 5. PUE baseline comparisons
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nIsolation Forest:")
print(f"  Anomalies: {ml_count:,}")
print(f"  Rate:      {ml_rate:.2%}")


print("\nPUE threshold baselines:")

for threshold in PUE_THRESHOLDS:

    baseline_anomalies = (
        model_df["pue"] >= threshold
    )

    count = int(
        baseline_anomalies.sum()
    )

    rate = (
        count / len(model_df)
    )

    print(
        f"  PUE >= {threshold:.2f} "
        f"→ {count:,} observations "
        f"({rate:.2%})"
    )


# ------------------------------------------------------------
# 6. Compare overlap
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ML / BASELINE OVERLAP")
print("=" * 60)

for threshold in PUE_THRESHOLDS:

    baseline = (
        model_df["pue"] >= threshold
    )

    overlap = (
        ml_anomalies & baseline
    ).sum()

    ml_only = (
        ml_anomalies & ~baseline
    ).sum()

    baseline_only = (
        baseline & ~ml_anomalies
    ).sum()

    print(
        f"\nPUE >= {threshold:.2f}"
    )

    print(
        f"  Both ML + baseline: {overlap:,}"
    )

    print(
        f"  ML only:            {ml_only:,}"
    )

    print(
        f"  Baseline only:      {baseline_only:,}"
    )


# ------------------------------------------------------------
# 7. Sustainability characteristics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SUSTAINABILITY CHARACTERISTICS")
print("=" * 60)

normal = model_df.loc[~ml_anomalies]

ml_anomaly_df = model_df.loc[ml_anomalies]

print("\nIsolation Forest anomalies:")

print(
    f"  Mean PUE: "
    f"{ml_anomaly_df['pue'].mean():.4f}"
)

print(
    f"  Median PUE: "
    f"{ml_anomaly_df['pue'].median():.4f}"
)

print(
    f"  Mean IT power: "
    f"{ml_anomaly_df['it_power_kw'].mean():.2f} kW"
)


print("\nNormal observations:")

print(
    f"  Mean PUE: "
    f"{normal['pue'].mean():.4f}"
)

print(
    f"  Median PUE: "
    f"{normal['pue'].median():.4f}"
)

print(
    f"  Mean IT power: "
    f"{normal['it_power_kw'].mean():.2f} kW"
)


# ------------------------------------------------------------
# 8. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    """
The PUE threshold is a simple operational baseline.

Isolation Forest considers multiple sustainability-related
features simultaneously, including IT power, cooling, HVAC,
pump behavior, PUE and subsystem-to-IT ratios.

This comparison does NOT establish that Isolation Forest is
more accurate, because there is no ground-truth failure label.

Instead, it evaluates whether ML identifies observations that
are different from what a simple PUE-only rule would capture.

A useful ML detector should identify some anomalies that a
PUE-only threshold misses, while still showing meaningful
sustainability characteristics.
"""
)

print("\nEvaluation completed successfully.")