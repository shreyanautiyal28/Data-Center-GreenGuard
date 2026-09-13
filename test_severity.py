import pandas as pd

from root_cause.analyzer import classify_root_cause
from validation.plausibility import validate_operational_plausibility
from risk.severity import calculate_severity


INPUT_PATH = "data/processed/anomaly_results.parquet"


print("========================================")
print("GREENGUARD SUSTAINABILITY PRIORITIZATION")
print("========================================")


# --------------------------------------------------
# Load anomalies
# --------------------------------------------------

print("\nLoading anomaly results...")

df = pd.read_parquet(INPUT_PATH)

anomalies = df[
    df["anomaly_label"] == -1
].copy()

print(
    "Total anomalies:",
    len(anomalies)
)


# --------------------------------------------------
# Root cause
# --------------------------------------------------

print("\nRunning root-cause analysis...")

anomalies = classify_root_cause(
    anomalies
)


# --------------------------------------------------
# Data-quality validation
# --------------------------------------------------

print("\nRunning plausibility validation...")

anomalies = validate_operational_plausibility(
    anomalies
)


# --------------------------------------------------
# Sustainability priority
# --------------------------------------------------

print("\nCalculating sustainability priority...")

anomalies = calculate_severity(
    anomalies
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n========================================")
print("SEVERITY DISTRIBUTION")
print("========================================")

print(
    anomalies["severity"].value_counts()
)


print("\n========================================")
print("DATA QUALITY DISTRIBUTION")
print("========================================")

print(
    anomalies["data_quality_status"].value_counts()
)


print("\n========================================")
print("TOP PRIORITY EVENTS")
print("========================================")


columns = [
    "ts",
    "it_power_kw",
    "pue",
    "estimated_excess_power_kw",
    "root_cause",
    "contributing_factors",
    "data_quality_status",
    "sustainability_priority_score",
    "severity"
]

available = [
    c for c in columns
    if c in anomalies.columns
]


top_events = anomalies.sort_values(
    "sustainability_priority_score",
    ascending=False
).head(20)


print(
    top_events[available]
    .to_string(index=False)
)


print("\nSustainability prioritization completed.")