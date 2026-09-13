import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import pandas as pd

from root_cause.analyzer import classify_root_cause
from validation.plausibility import validate_operational_plausibility
from risk.severity import calculate_severity


INPUT_PATH = "data/processed/anomaly_results.parquet"
OUTPUT_PATH = "data/processed/greenguard_analyzed_anomalies.parquet"


print("=" * 60)
print("GREENGUARD FINAL ANOMALY ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load anomaly results
# ---------------------------------------------------------

print("\nLoading anomaly results...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows loaded: {len(df):,}")

# Keep only detected anomalies
if "anomaly_label" not in df.columns:
    raise ValueError(
        "Could not find 'anomaly_label' in anomaly results."
    )

anomalies = df[df["anomaly_label"] == -1].copy()

print(f"Anomalies selected: {len(anomalies):,}")


# ---------------------------------------------------------
# 2. Root-cause analysis
# ---------------------------------------------------------

print("\nRunning root-cause analysis...")

anomalies = classify_root_cause(anomalies)


# ---------------------------------------------------------
# 3. Operational plausibility validation
# ---------------------------------------------------------

print("\nRunning plausibility validation...")

anomalies = validate_operational_plausibility(anomalies)


# ---------------------------------------------------------
# 4. Sustainability prioritization
# ---------------------------------------------------------

print("\nCalculating sustainability priority...")

anomalies = calculate_severity(anomalies)


# ---------------------------------------------------------
# 5. Sort by priority
# ---------------------------------------------------------

anomalies = anomalies.sort_values(
    by="sustainability_priority_score",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# 6. Save final artifact
# ---------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

anomalies.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# 7. Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

print(f"\nOutput:")
print(OUTPUT_PATH)

print(f"\nTotal analyzed anomalies: {len(anomalies):,}")


print("\n" + "=" * 60)
print("SEVERITY DISTRIBUTION")
print("=" * 60)

print(
    anomalies["severity"]
    .value_counts()
)


print("\n" + "=" * 60)
print("DATA QUALITY DISTRIBUTION")
print("=" * 60)

print(
    anomalies["data_quality_status"]
    .value_counts()
)


print("\n" + "=" * 60)
print("ROOT-CAUSE DISTRIBUTION")
print("=" * 60)

print(
    anomalies["root_cause"]
    .value_counts()
)


print("\n" + "=" * 60)
print("TOP 10 PRIORITY EVENTS")
print("=" * 60)

columns = [
    "ts",
    "it_power_kw",
    "pue",
    "root_cause",
    "data_quality_status",
    "sustainability_priority_score",
    "severity",
]

available_columns = [
    col for col in columns
    if col in anomalies.columns
]

print(
    anomalies[available_columns]
    .head(10)
    .to_string(index=False)
)

print("\nFinal GreenGuard analysis completed successfully.")