import pandas as pd

from impact.calculator import (
    calculate_sustainability_impact
)


INPUT_PATH = (
    "data/processed/anomaly_results.parquet"
)


print("Loading anomaly results...")

df = pd.read_parquet(INPUT_PATH)

print(
    f"Rows loaded: {len(df):,}"
)


# ----------------------------------------------------------
# Calculate impact
# ----------------------------------------------------------

df = calculate_sustainability_impact(df)


# ----------------------------------------------------------
# Analyze only ML anomalies
# ----------------------------------------------------------

anomalies = df[
    df["anomaly_label"] == -1
].copy()


print("\n========================================")
print("GREENGUARD SUSTAINABILITY IMPACT")
print("========================================")


print(
    "\nTotal anomalies:",
    len(anomalies)
)


print(
    "\nBaseline PUE:",
    round(
        df["baseline_pue"].iloc[0],
        4
    )
)


print(
    "\nEstimated excess power:"
)


print(
    anomalies[
        "estimated_excess_power_kw"
    ].describe()
)


print(
    "\nTop potential sustainability-impact events:"
)


columns = [
    "ts",
    "it_power_kw",
    "pue",
    "baseline_pue",
    "estimated_excess_power_kw",
    "cooling_to_it_ratio",
    "anomaly_score"
]


print(
    anomalies[
        columns
    ]
    .sort_values(
        "estimated_excess_power_kw",
        ascending=False
    )
    .head(10)
)


print(
    "\nImpact analysis completed."
)
