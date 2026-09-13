import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD CONTROLLED ANOMALY VALIDATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/esif_pue_features.parquet"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


print("=" * 65)
print("GREENGUARD CONTROLLED ANOMALY VALIDATION")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load real NLR data
# ------------------------------------------------------------

print("\nLoading real NLR/ESIF feature data...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")


if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

print(f"Rows selected: {len(df):,}")


# ------------------------------------------------------------
# 2. Required features
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

missing = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required features: {missing}"
    )


# ------------------------------------------------------------
# 3. Clean base data
# ------------------------------------------------------------

base = df.copy()

base = base.replace(
    [np.inf, -np.inf],
    np.nan
)

base = base.dropna(
    subset=FEATURE_COLUMNS
).reset_index(drop=True)

print(
    f"Rows available for validation: {len(base):,}"
)


# ------------------------------------------------------------
# 4. Select normal operational observations
# ------------------------------------------------------------

# We deliberately avoid obviously problematic observations
# when selecting the base population for controlled injection.

normal_mask = (
    (base["it_power_kw"] > 100)
    & (base["pue"] > 0.95)
    & (base["pue"] < 1.15)
    & (base["cooling_to_it_ratio"] < 0.20)
    & (base["hvac_to_it_ratio"] < 0.20)
    & (base["pump_to_it_ratio"] < 0.15)
)

normal = base[normal_mask].copy()

print(
    f"Normal candidate observations: {len(normal):,}"
)


if len(normal) < 1000:
    raise ValueError(
        "Not enough normal observations available "
        "for controlled validation."
    )


# ------------------------------------------------------------
# 5. Create controlled anomaly scenarios
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

N_PER_SCENARIO = 100


# Select distinct rows for each scenario
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

cooling_events["cooling_kw"] *= 3.0
cooling_events["injected_type"] = "Cooling inefficiency"


# ------------------------------------------------------------
# Scenario B: HVAC inefficiency
# ------------------------------------------------------------

hvac_events["hvac_kw"] *= 3.0
hvac_events["injected_type"] = "HVAC inefficiency"


# ------------------------------------------------------------
# Scenario C: Pump inefficiency
# ------------------------------------------------------------

pump_events["pump_kw"] *= 3.0
pump_events["injected_type"] = "Pump inefficiency"


injected = pd.concat(
    [
        cooling_events,
        hvac_events,
        pump_events
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 6. Recalculate derived sustainability features
# ------------------------------------------------------------

EPSILON = 1e-6

injected["non_it_power_kw"] = (
    injected["cooling_kw"]
    + injected["hvac_kw"]
    + injected["pump_kw"]
    + injected["plug_and_light_kw"]
)

injected["cooling_to_it_ratio"] = (
    injected["cooling_kw"]
    / injected["it_power_kw"].clip(lower=EPSILON)
)

injected["hvac_to_it_ratio"] = (
    injected["hvac_kw"]
    / injected["it_power_kw"].clip(lower=EPSILON)
)

injected["pump_to_it_ratio"] = (
    injected["pump_kw"]
    / injected["it_power_kw"].clip(lower=EPSILON)
)

injected["non_it_to_it_ratio"] = (
    injected["non_it_power_kw"]
    / injected["it_power_kw"].clip(lower=EPSILON)
)

injected["pue_proxy"] = (
    1
    + injected["non_it_power_kw"]
    / injected["it_power_kw"].clip(lower=EPSILON)
)


# Use the calculated proxy as the injected PUE
injected["pue"] = injected["pue_proxy"]


# ------------------------------------------------------------
# 7. Prepare ML matrix
# ------------------------------------------------------------

X_normal = normal[
    FEATURE_COLUMNS
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

X_injected = injected[
    FEATURE_COLUMNS
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


# ------------------------------------------------------------
# 8. Train detector on normal real observations
# ------------------------------------------------------------

print("\nTraining Isolation Forest on real operational data...")

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
# 9. Detect injected anomalies
# ------------------------------------------------------------

X_test = scaler.transform(
    X_injected
)

predictions = model.predict(
    X_test
)

injected["detected"] = (
    predictions == -1
)


# ------------------------------------------------------------
# 10. Overall detection rate
# ------------------------------------------------------------

overall_detection = (
    injected["detected"].mean()
)

print("\n" + "=" * 65)
print("OVERALL DETECTION")
print("=" * 65)

print(
    f"Injected anomalies: {len(injected):,}"
)

print(
    f"Detected anomalies: "
    f"{injected['detected'].sum():,}"
)

print(
    f"Detection rate: "
    f"{overall_detection:.2%}"
)


# ------------------------------------------------------------
# 11. Detection by scenario
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("DETECTION BY ANOMALY TYPE")
print("=" * 65)

scenario_results = (
    injected
    .groupby("injected_type")["detected"]
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
# 12. Compare feature changes
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("INJECTED EVENT CHARACTERISTICS")
print("=" * 65)

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


# ------------------------------------------------------------
# 13. Save validation results
# ------------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "controlled_anomaly_validation.parquet"
)

injected.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# 14. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

print(
    """
This experiment injects controlled sustainability anomalies
into observations sampled from real NLR/ESIF operational data.

The injected anomalies are NOT presented as real-world events.

They are used only to test whether the anomaly detector can
identify known perturbations.

Detection rate here is therefore a controlled validation
metric, not a claim of real-world failure-detection accuracy.

The real NLR/ESIF observations remain the primary evidence
for the GreenGuard analysis.
"""
)

print(
    f"\nValidation artifact saved to:\n{OUTPUT_PATH}"
)

print("\nControlled validation completed successfully.")