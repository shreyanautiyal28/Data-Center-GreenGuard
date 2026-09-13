import pandas as pd
import numpy as np

INPUT = "data/processed/anomaly_v2_validation.parquet"
OUTPUT = "data/processed/final_v2_impact_validation.parquet"

print("=" * 70)
print("GREENGUARD FINAL V2 ANOMALY + SUSTAINABILITY IMPACT VALIDATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------
print("\nLoading V2 validation data...")

df = pd.read_parquet(INPUT)

print(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------------
# 2. VALID MODELING ROWS
# ---------------------------------------------------------
required = [
    "it_power_kw",
    "pue",
    "expected_hvac_kw",
    "expected_pump_kw",
    "hvac_kw",
    "pump_kw",
    "cooling_kw",
    "expected_cooling_kw",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "cooling_deviation_ratio",
    "anomaly_label_v2",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()

print(f"Rows after validation: {len(df):,}")

# ---------------------------------------------------------
# 3. ANOMALY FLAG
# ---------------------------------------------------------
df["is_anomaly_v2"] = (
    df["anomaly_label_v2"] == -1
).astype(int)

print("\nAnomaly distribution:")
print(df["is_anomaly_v2"].value_counts().sort_index())

# ---------------------------------------------------------
# 4. PUE BASELINE
# ---------------------------------------------------------
normal = df[df["is_anomaly_v2"] == 0]

if len(normal) == 0:
    raise ValueError("No normal observations available.")

baseline_pue = normal["pue"].median()

print(f"\nNormal-observation baseline PUE: {baseline_pue:.4f}")

# ---------------------------------------------------------
# 5. ESTIMATED EXCESS POWER
# ---------------------------------------------------------
#
# Assumption:
#
#   excess power =
#       IT power × (observed PUE - baseline PUE)
#
# This is an ESTIMATE, not measured energy waste.
#
# Negative values are clipped to zero because the metric
# represents potential excess consumption relative to baseline.
# ---------------------------------------------------------

df["pue_excess"] = (
    df["pue"] - baseline_pue
).clip(lower=0)

df["estimated_excess_power_kw"] = (
    df["it_power_kw"] *
    df["pue_excess"]
)

# ---------------------------------------------------------
# 6. ESTIMATED EXCESS ENERGY
# ---------------------------------------------------------
#
# Dataset observations are approximately minute-level,
# but timestamps are not assumed to be perfectly uniform.
#
# Therefore we DO NOT blindly multiply by 1 hour.
#
# Instead calculate duration from consecutive timestamps.
# ---------------------------------------------------------

df = df.sort_values("ts").reset_index(drop=True)

df["interval_hours"] = (
    df["ts"].shift(-1) - df["ts"]
).dt.total_seconds() / 3600

# Keep only realistic positive intervals.
df.loc[
    (df["interval_hours"] <= 0) |
    (df["interval_hours"] > 1),
    "interval_hours"
] = np.nan

df["estimated_excess_energy_kwh"] = (
    df["estimated_excess_power_kw"] *
    df["interval_hours"]
)

# ---------------------------------------------------------
# 7. OPERATIONAL PLAUSIBILITY
# ---------------------------------------------------------

df["low_it_load"] = (
    df["it_power_kw"] <= 10
)

df["extreme_pue"] = (
    df["pue"] > 3
)

df["extreme_hvac_ratio"] = (
    df["hvac_to_it_ratio"] > 1
)

df["extreme_pump_ratio"] = (
    df["pump_to_it_ratio"] > 1
)

df["extreme_cooling_ratio"] = (
    df["cooling_to_it_ratio"] > 1
)

df["review_required"] = (
    df[
        [
            "low_it_load",
            "extreme_pue",
            "extreme_hvac_ratio",
            "extreme_pump_ratio",
            "extreme_cooling_ratio",
        ]
    ]
    .any(axis=1)
)

df["plausibility_status"] = np.where(
    df["review_required"],
    "Review-required",
    "Operationally-plausible"
)

# ---------------------------------------------------------
# 8. IMPACT-RELEVANT SIGNAL
# ---------------------------------------------------------

df["impact_signal"] = (
    df["estimated_excess_power_kw"] > 0
).astype(int)

# ---------------------------------------------------------
# 9. ANOMALY IMPACT SUMMARY
# ---------------------------------------------------------

anomalies = df[
    df["is_anomaly_v2"] == 1
].copy()

print("\n" + "=" * 70)
print("FINAL V2 RESULTS")
print("=" * 70)

print(f"\nTotal modeling observations : {len(df):,}")
print(f"V2 anomalies                : {len(anomalies):,}")
print(
    f"Anomaly rate                : "
    f"{len(anomalies) / len(df) * 100:.2f}%"
)

print("\nAnomaly PUE:")
print(anomalies["pue"].describe())

print("\nAnomaly HVAC deviation ratio:")
print(anomalies["hvac_deviation_ratio"].describe())

print("\nAnomaly pump deviation ratio:")
print(anomalies["pump_deviation_ratio"].describe())

print("\nAnomaly cooling deviation ratio:")
print(anomalies["cooling_deviation_ratio"].describe())

print("\nEstimated excess power:")
print(
    anomalies["estimated_excess_power_kw"].describe()
)

print("\nEstimated excess energy:")
print(
    anomalies["estimated_excess_energy_kwh"].describe()
)

# ---------------------------------------------------------
# 10. HIGH-IMPACT ANOMALIES
# ---------------------------------------------------------

high_impact = anomalies.sort_values(
    "estimated_excess_power_kw",
    ascending=False
)

print("\nTop 10 potential sustainability-impact observations:")

cols = [
    "ts",
    "it_power_kw",
    "pue",
    "estimated_excess_power_kw",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "cooling_deviation_ratio",
    "plausibility_status",
]

print(
    high_impact[cols]
    .head(10)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 11. PLAUSIBILITY SUMMARY
# ---------------------------------------------------------

print("\nPlausibility distribution:")

print(
    anomalies["plausibility_status"]
    .value_counts()
)

# ---------------------------------------------------------
# 12. IMPACT SIGNAL SUMMARY
# ---------------------------------------------------------

print("\nPositive estimated excess-power observations:")

print(
    anomalies["impact_signal"]
    .value_counts()
)

# ---------------------------------------------------------
# 13. SAVE
# ---------------------------------------------------------

df.to_parquet(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)