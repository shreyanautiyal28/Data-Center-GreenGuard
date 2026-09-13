import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GREENGUARD ANOMALY MAGNITUDE SENSITIVITY
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/esif_pue_features.parquet"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

MULTIPLIERS = [1.25, 1.5, 2.0, 3.0, 5.0]

N_PER_SCENARIO = 100


print("=" * 65)
print("GREENGUARD ANOMALY MAGNITUDE SENSITIVITY")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load real data
# ------------------------------------------------------------

print("\nLoading real NLR/ESIF data...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows available: {len(df):,}")


if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ------------------------------------------------------------
# 2. Features
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
        f"Missing features: {missing}"
    )


# ------------------------------------------------------------
# 3. Clean data
# ------------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURE_COLUMNS
).reset_index(drop=True)


# ------------------------------------------------------------
# 4. Establish normal population
# ------------------------------------------------------------

normal_mask = (
    (df["it_power_kw"] > 100)
    & (df["pue"] > 0.95)
    & (df["pue"] < 1.15)
    & (df["cooling_to_it_ratio"] < 0.20)
    & (df["hvac_to_it_ratio"] < 0.20)
    & (df["pump_to_it_ratio"] < 0.15)
)

normal = df[normal_mask].copy()

print(
    f"Normal candidate observations: {len(normal):,}"
)


# ------------------------------------------------------------
# 5. Train model on real operational data
# ------------------------------------------------------------

X_train = normal[
    FEATURE_COLUMNS
].values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)


model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_scaled)


# ------------------------------------------------------------
# 6. Run magnitude experiments
# ------------------------------------------------------------

results = []


for subsystem in [
    "cooling_kw",
    "hvac_kw",
    "pump_kw"
]:

    for multiplier in MULTIPLIERS:

        sample = normal.sample(
            n=N_PER_SCENARIO,
            random_state=(
                RANDOM_STATE
                + int(multiplier * 100)
                + len(subsystem)
            )
        ).copy()


        # Apply controlled perturbation
        sample[subsystem] *= multiplier


        # Recalculate derived features
        sample["non_it_power_kw"] = (
            sample["cooling_kw"]
            + sample["hvac_kw"]
            + sample["pump_kw"]
            + sample["plug_and_light_kw"]
        )


        epsilon = 1e-6


        sample["cooling_to_it_ratio"] = (
            sample["cooling_kw"]
            / sample["it_power_kw"].clip(
                lower=epsilon
            )
        )


        sample["hvac_to_it_ratio"] = (
            sample["hvac_kw"]
            / sample["it_power_kw"].clip(
                lower=epsilon
            )
        )


        sample["pump_to_it_ratio"] = (
            sample["pump_kw"]
            / sample["it_power_kw"].clip(
                lower=epsilon
            )
        )


        sample["non_it_to_it_ratio"] = (
            sample["non_it_power_kw"]
            / sample["it_power_kw"].clip(
                lower=epsilon
            )
        )


        sample["pue_proxy"] = (
            1
            + sample["non_it_power_kw"]
            / sample["it_power_kw"].clip(
                lower=epsilon
            )
        )


        sample["pue"] = sample["pue_proxy"]


        # Predict
        X_test = sample[
            FEATURE_COLUMNS
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()


        predictions = model.predict(
            scaler.transform(
                X_test
            )
        )


        detection_rate = (
            predictions == -1
        ).mean()


        results.append(
            {
                "subsystem": subsystem,
                "multiplier": multiplier,
                "detected": int(
                    (predictions == -1).sum()
                ),
                "tested": len(predictions),
                "detection_rate": detection_rate,
                "mean_pue": sample["pue"].mean(),
                "mean_ratio": (
                    sample[subsystem.replace(
                        "_kw", ""
                    ) + "_to_it_ratio"].mean()
                    if subsystem != "cooling_kw"
                    else sample[
                        "cooling_to_it_ratio"
                    ].mean()
                )
            }
        )


# ------------------------------------------------------------
# 7. Results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)


print("\n" + "=" * 65)
print("DETECTION SENSITIVITY RESULTS")
print("=" * 65)

print(
    results_df[
        [
            "subsystem",
            "multiplier",
            "detected",
            "tested",
            "detection_rate",
            "mean_pue",
            "mean_ratio",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "anomaly_sensitivity_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 65)
print("VALIDATION INTERPRETATION")
print("=" * 65)

print(
    """
This experiment evaluates whether detection sensitivity
increases as controlled subsystem inefficiency becomes
more pronounced.

The perturbations are applied to observations sampled from
real NLR/ESIF operational data.

They are controlled validation cases, not real operational
events.

The results should NOT be interpreted as production
accuracy or real-world failure detection performance.
"""
)

print(
    f"\nResults saved to:\n{OUTPUT_PATH}"
)

print("\nSensitivity evaluation completed.")