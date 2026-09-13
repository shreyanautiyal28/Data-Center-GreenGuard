import pandas as pd

from impact.calculator import (
    calculate_sustainability_impact
)

from risk.severity import (
    calculate_severity
)


INPUT_PATH = (
    "data/processed/anomaly_results.parquet"
)


print("Loading anomaly results...")

df = pd.read_parquet(INPUT_PATH)


# ==========================================================
# Calculate sustainability impact
# ==========================================================

df = calculate_sustainability_impact(df)


# ==========================================================
# Calculate severity
# ==========================================================

df = calculate_severity(df)


# ==========================================================
# Keep ML anomalies
# ==========================================================

anomalies = df[
    df["anomaly_label"] == -1
].copy()


print("\n========================================")
print("GREENGUARD RISK ENGINE")
print("========================================")


print(
    "\nAnomalies:",
    len(anomalies)
)


print("\nSeverity distribution:")

print(
    anomalies["severity"]
    .value_counts()
    .sort_index()
)


print("\nRisk score statistics:")

print(
    anomalies["risk_score"]
    .describe()
)


print("\nTop GreenGuard alerts:")

columns = [
    "ts",
    "it_power_kw",
    "pue",
    "cooling_to_it_ratio",
    "estimated_excess_power_kw",
    "anomaly_score",
    "risk_score",
    "severity"
]


print(
    anomalies[
        columns
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(15)
)


print("\nRisk analysis completed.")