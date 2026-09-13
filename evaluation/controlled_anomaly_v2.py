import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD CONTROLLED ANOMALY VALIDATION - V2
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/rolling_baseline_validation.parquet"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

N_PER_SCENARIO = 100
MULTIPLIER = 3.0


print("=" * 70)
print("GREENGUARD CONTROLLED ANOMALY VALIDATION - V2")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load rolling-baseline validation data
# ------------------------------------------------------------

print("\nLoading rolling-baseline validation data...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

print(f"Rows selected: {len(df):,}")


# ------------------------------------------------------------
# 2. Required columns
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue",
    "expected_hvac_kw",
    "expected_pump_kw",
]

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ------------------------------------------------------------
# 3. Clean data
# ------------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# 4. Reconstruct V2 baseline features
# ------------------------------------------------------------

EPSILON = 1e-6

safe_it = df["it_power_kw"].clip(
    lower=EPSILON
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

df["non_it_power_kw"] = (
    df["cooling_kw"]
    + df["hvac_kw"]
    + df["pump_kw"]
)

df["non_it_to_it_ratio"] = (
    df["non_it_power_kw"] / safe_it
)


# ------------------------------------------------------------
# 5. Expected-behavior deviation features
# ------------------------------------------------------------

df["hvac_deviation_ratio"] = (
    (
        df["hvac_kw"]
        - df["expected_hvac_kw"]
    )
    / df["expected_hvac_kw"].clip(
        lower=EPSILON
    )
)

df["pump_deviation_ratio"] = (
    (
        df["pump_kw"]
        - df["expected_pump_kw"]
    )
    / df["expected_pump_kw"].clip(
        lower=EPSILON
    )
)


# ------------------------------------------------------------
# 6. Select normal operational observations
# ------------------------------------------------------------

normal_mask = (
    (df["it_power_kw"] > 100)
    & (df["pue"] > 0.95)
    & (df["pue"] < 1.15)
    & (df["cooling_to_it_ratio"] < 0.20)
    & (df["hvac_to_it_ratio"] < 0.20)
    & (df["pump_to_it_ratio"] < 0.15)
    & (df["expected_hvac_kw"] > 1)
    & (df["expected_pump_kw"] > 1)
)

normal = df[normal_mask].copy()

print(
    f"Normal candidate observations: "
    f"{len(normal):,}"
)

if len(normal) < 1000:
    raise ValueError(
        "Not enough normal observations available "
        "for controlled validation."
    )


# ------------------------------------------------------------
# 7. Create controlled anomaly scenarios
# ------------------------------------------------------------

selected = normal.sample(
    n=N_PER_SCENARIO * 3,
    random_state=RANDOM_STATE
).copy()


cooling_events = selected.iloc[
    0:N_PER_SCENARIO
].copy()

hvac_events = selected.iloc[
    N_PER_SCENARIO:2 * N_PER_SCENARIO
].copy()

pump_events = selected.iloc[
    2 * N_PER_SCENARIO:3 * N_PER_SCENARIO
].copy()


# ------------------------------------------------------------
# Scenario A: Cooling inefficiency
# ------------------------------------------------------------

cooling_events["cooling_kw"] *= MULTIPLIER
cooling_events["injected_type"] = "Cooling inefficiency"


# ------------------------------------------------------------
# Scenario B: HVAC inefficiency
# ------------------------------------------------------------

hvac_events["hvac_kw"] *= MULTIPLIER
hvac_events["injected_type"] = "HVAC inefficiency"


# ------------------------------------------------------------
# Scenario C: Pump inefficiency
# ------------------------------------------------------------

pump_events["pump_kw"] *= MULTIPLIER
pump_events["injected_type"] = "Pump inefficiency"


# ------------------------------------------------------------
# Combine scenarios
# ------------------------------------------------------------

injected = pd.concat(
    [
        cooling_events,
        hvac_events,
        pump_events
    ],
    ignore_index=True
)


print(
    f"Controlled anomalies created: "
    f"{len(injected):,}"
)


# ------------------------------------------------------------
# 8. Recalculate derived features after injection
# ------------------------------------------------------------

safe_it = injected["it_power_kw"].clip(
    lower=EPSILON
)

injected["non_it_power_kw"] = (
    injected["cooling_kw"]
    + injected["hvac_kw"]
    + injected["pump_kw"]
)

injected["cooling_to_it_ratio"] = (
    injected["cooling_kw"] / safe_it
)

injected["hvac_to_it_ratio"] = (
    injected["hvac_kw"] / safe_it
)

injected["pump_to_it_ratio"] = (
    injected["pump_kw"] / safe_it
)

injected["non_it_to_it_ratio"] = (
    injected["non_it_power_kw"] / safe_it
)


# ------------------------------------------------------------
# 9. Recalculate injected PUE
# ------------------------------------------------------------

injected["pue_proxy"] = (
    1
    + injected["non_it_power_kw"] / safe_it
)

injected["pue"] = injected["pue_proxy"]


# ------------------------------------------------------------
# 10. Recalculate expected-behavior deviations
# ------------------------------------------------------------

injected["hvac_deviation_ratio"] = (
    (
        injected["hvac_kw"]
        - injected["expected_hvac_kw"]
    )
    / injected["expected_hvac_kw"].clip(
        lower=EPSILON
    )
)

injected["pump_deviation_ratio"] = (
    (
        injected["pump_kw"]
        - injected["expected_pump_kw"]
    )
    / injected["expected_pump_kw"].clip(
        lower=EPSILON
    )
)


# ------------------------------------------------------------
# 11. V2 feature set
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 12. Prepare ML matrices
# ------------------------------------------------------------

X_normal = (
    normal[FEATURE_COLUMNS_V2]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

X_injected = (
    injected[FEATURE_COLUMNS_V2]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)


print(
    f"Normal training rows: "
    f"{len(X_normal):,}"
)

print(
    f"Injected test rows: "
    f"{len(X_injected):,}"
)


# ------------------------------------------------------------
# 13. Train V2 Isolation Forest
# ------------------------------------------------------------

print(
    "\nTraining V2 Isolation Forest "
    "on real operational data..."
)

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_normal
)

model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)


# ------------------------------------------------------------
# 14. Detect controlled anomalies
# ------------------------------------------------------------

X_test = scaler.transform(
    X_injected
)

predictions = model.predict(
    X_test
)

injected["detected_v2"] = (
    predictions == -1
)


# ------------------------------------------------------------
# 15. Overall detection
# ------------------------------------------------------------

overall_detection = (
    injected["detected_v2"].mean()
)

print("\n" + "=" * 70)
print("V2 OVERALL DETECTION")
print("=" * 70)

print(
    f"Injected anomalies: "
    f"{len(injected):,}"
)

print(
    f"Detected anomalies: "
    f"{injected['detected_v2'].sum():,}"
)

print(
    f"Detection rate: "
    f"{overall_detection:.2%}"
)


# ------------------------------------------------------------
# 16. Detection by anomaly type
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V2 DETECTION BY ANOMALY TYPE")
print("=" * 70)

scenario_results = (
    injected
    .groupby("injected_type")["detected_v2"]
    .agg(
        injected_count="count",
        detected_count="sum",
        detection_rate="mean"
    )
)

print(
    scenario_results.to_string()
)


# ------------------------------------------------------------
# 17. Injected event characteristics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V2 INJECTED EVENT CHARACTERISTICS")
print("=" * 70)

for scenario in injected["injected_type"].unique():

    subset = injected[
        injected["injected_type"] == scenario
    ]

    print(f"\n{scenario}")

    print(
        f"  Mean PUE: "
        f"{subset['pue'].mean():.3f}"
    )

    print(
        f"  Mean cooling/IT ratio: "
        f"{subset['cooling_to_it_ratio'].mean():.3f}"
    )

    print(
        f"  Mean HVAC/IT ratio: "
        f"{subset['hvac_to_it_ratio'].mean():.3f}"
    )

    print(
        f"  Mean pump/IT ratio: "
        f"{subset['pump_to_it_ratio'].mean():.3f}"
    )

    print(
        f"  Mean HVAC deviation: "
        f"{subset['hvac_deviation_ratio'].mean():.3f}"
    )

    print(
        f"  Mean pump deviation: "
        f"{subset['pump_deviation_ratio'].mean():.3f}"
    )


# ------------------------------------------------------------
# 18. Save validation artifact
# ------------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "controlled_anomaly_v2_validation.parquet"
)

injected.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# 19. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This experiment evaluates the GreenGuard V2 anomaly detector
using controlled perturbations applied to real NLR/ESIF
operational observations.

The experiment uses the same sampling strategy and anomaly
injection methodology as the original V1 controlled validation.

V2 additionally incorporates expected-behavior deviation
features for HVAC and pump subsystems.

The injected anomalies are NOT real-world failures.

Detection rate is therefore a controlled validation metric,
not a claim of real-world failure-detection accuracy.

The real NLR/ESIF observations remain the primary evidence
for GreenGuard's operational sustainability analysis.
"""
)

print(
    f"\nValidation artifact saved to:\n{OUTPUT_PATH}"
)

print(
    "\nControlled V2 validation completed successfully."
)