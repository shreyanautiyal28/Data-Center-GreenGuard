import pandas as pd

from root_cause.analyzer import (
    classify_root_cause
)


INPUT_PATH = (
    "data/processed/anomaly_results.parquet"
)


print("Loading anomaly results...")

df = pd.read_parquet(INPUT_PATH)


# ----------------------------------------------------------
# Keep ML anomalies
# ----------------------------------------------------------

anomalies = df[
    df["anomaly_label"] == -1
].copy()


# ----------------------------------------------------------
# Root-cause classification
# ----------------------------------------------------------

anomalies = classify_root_cause(
    anomalies
)


print("\n========================================")
print("GREENGUARD ROOT-CAUSE ANALYSIS")
print("========================================")


print(
    "\nTotal anomalies:",
    len(anomalies)
)


print(
    "\nRoot-cause distribution:"
)


print(
    anomalies["root_cause"]
    .value_counts()
)


print(
    "\nTop events with explanations:"
)


columns = [
    "ts",
    "it_power_kw",
    "pue",
    "cooling_to_it_ratio",
    "anomaly_score",
    "root_cause",
    "root_cause_evidence"
]


available = [
    c
    for c in columns
    if c in anomalies.columns
]


print(
    anomalies[
        available
    ].head(20).to_string(
        index=False
    )
)


print(
    "\nRoot-cause analysis completed."
)