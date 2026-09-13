import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD V2 STABILITY VALIDATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/rolling_baseline_validation.parquet"

SEEDS = [42, 52, 62, 72, 82]

N_PER_SCENARIO = 100
MULTIPLIER = 3.0

EPSILON = 1e-6


print("=" * 70)
print("GREENGUARD V2 STABILITY VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\nLoading rolling-baseline validation data...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")


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
# 3. Clean
# ------------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# 4. Build V2 features
# ------------------------------------------------------------

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
# 5. Select normal population
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


# ------------------------------------------------------------
# 6. V2 feature set
# ------------------------------------------------------------

FEATURE_COLUMNS = [
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
# 7. Run multiple seeds
# ------------------------------------------------------------

all_results = []


for seed in SEEDS:

    print("\n" + "-" * 70)
    print(f"RUNNING SEED: {seed}")
    print("-" * 70)

    # --------------------------------------------
    # Select 300 distinct normal observations
    # --------------------------------------------

    selected = normal.sample(
        n=N_PER_SCENARIO * 3,
        random_state=seed
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


    # --------------------------------------------
    # Inject anomalies
    # --------------------------------------------

    cooling_events["cooling_kw"] *= MULTIPLIER
    cooling_events["injected_type"] = (
        "Cooling inefficiency"
    )

    hvac_events["hvac_kw"] *= MULTIPLIER
    hvac_events["injected_type"] = (
        "HVAC inefficiency"
    )

    pump_events["pump_kw"] *= MULTIPLIER
    pump_events["injected_type"] = (
        "Pump inefficiency"
    )


    injected = pd.concat(
        [
            cooling_events,
            hvac_events,
            pump_events
        ],
        ignore_index=True
    )


    # --------------------------------------------
    # Recalculate features
    # --------------------------------------------

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

    injected["pue"] = (
        1
        + injected["non_it_power_kw"] / safe_it
    )

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


    # --------------------------------------------
    # Training data
    # --------------------------------------------

    X_normal = (
        normal[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    X_injected = (
        injected[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    # --------------------------------------------
    # Train V2
    # --------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_normal
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train)


    # --------------------------------------------
    # Detect
    # --------------------------------------------

    X_test = scaler.transform(
        X_injected
    )

    predictions = model.predict(
        X_test
    )

    injected["detected"] = (
        predictions == -1
    )


    # --------------------------------------------
    # Calculate results
    # --------------------------------------------

    overall = injected["detected"].mean()

    print(
        f"Overall detection: {overall:.2%}"
    )


    for scenario in [
        "Cooling inefficiency",
        "HVAC inefficiency",
        "Pump inefficiency"
    ]:

        subset = injected[
            injected["injected_type"] == scenario
        ]

        rate = subset["detected"].mean()

        print(
            f"{scenario}: {rate:.2%}"
        )

        all_results.append(
            {
                "seed": seed,
                "scenario": scenario,
                "detection_rate": rate
            }
        )


# ------------------------------------------------------------
# 8. Results table
# ------------------------------------------------------------

results = pd.DataFrame(all_results)


print("\n" + "=" * 70)
print("V2 STABILITY RESULTS")
print("=" * 70)

pivot = results.pivot(
    index="seed",
    columns="scenario",
    values="detection_rate"
)

print(
    (pivot * 100).round(2).to_string()
)


# ------------------------------------------------------------
# 9. Mean and standard deviation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MEAN ± STANDARD DEVIATION")
print("=" * 70)

summary = (
    results
    .groupby("scenario")["detection_rate"]
    .agg(
        mean="mean",
        std="std",
        minimum="min",
        maximum="max"
    )
)

summary_display = summary.copy()

for col in [
    "mean",
    "std",
    "minimum",
    "maximum"
]:
    summary_display[col] *= 100

summary_display = summary_display.round(2)

print(
    summary_display.to_string()
)


# ------------------------------------------------------------
# 10. Overall stability
# ------------------------------------------------------------

overall_by_seed = (
    results
    .groupby("seed")["detection_rate"]
    .mean()
)

print("\n" + "=" * 70)
print("OVERALL V2 STABILITY")
print("=" * 70)

print(
    f"Mean detection: "
    f"{overall_by_seed.mean():.2%}"
)

print(
    f"Std deviation: "
    f"{overall_by_seed.std():.2%}"
)

print(
    f"Minimum: "
    f"{overall_by_seed.min():.2%}"
)

print(
    f"Maximum: "
    f"{overall_by_seed.max():.2%}"
)


# ------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "anomaly_stability_v2.csv"
)

summary.to_csv(
    OUTPUT_PATH
)


# ------------------------------------------------------------
# 12. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This experiment evaluates whether V2 controlled anomaly
detection remains consistent across multiple random seeds.

The test uses real NLR/ESIF operational observations and
controlled perturbations.

Detection rates are validation metrics only and do not
represent real-world failure-detection accuracy.

Low variation across seeds increases confidence that the
observed V2 behavior is not caused by one particular sample.

Cooling, HVAC, and pump performance are evaluated separately.
"""
)

print(
    f"\nResults saved to:\n{OUTPUT_PATH}"
)

print(
    "\nV2 stability validation completed successfully."
)