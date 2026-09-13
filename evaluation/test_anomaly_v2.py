import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from models.anomaly_detector_v2 import (
    train_anomaly_detector_v2
)


INPUT_PATH = (
    "data/processed/"
    "rolling_baseline_validation.parquet"
)

OUTPUT_PATH = (
    "data/processed/"
    "anomaly_v2_validation.parquet"
)


print("=" * 70)
print("GREENGUARD ANOMALY DETECTOR V2")
print("=" * 70)


# ==========================================================
# LOAD
# ==========================================================

print("\nLoading walk-forward baseline results...")

df = pd.read_parquet(
    INPUT_PATH
)

print(
    f"Rows loaded: {len(df):,}"
)


# ==========================================================
# TRAIN V2
# ==========================================================

print(
    "\nTraining Isolation Forest V2..."
)

model, scaler, results = (
    train_anomaly_detector_v2(
        df,
        contamination=0.02
    )
)


# ==========================================================
# RESULTS
# ==========================================================

total = len(results)

anomalies = (
    results["anomaly_label_v2"] == -1
).sum()

normal = (
    results["anomaly_label_v2"] == 1
).sum()


print(
    "\n" + "=" * 70
)

print(
    "V2 ANOMALY RESULTS"
)

print(
    "=" * 70
)

print(
    f"Modeling rows : {total:,}"
)

print(
    f"Normal        : {normal:,}"
)

print(
    f"Anomalies     : {anomalies:,}"
)

print(
    f"Anomaly rate  : "
    f"{anomalies / total * 100:.2f}%"
)


# ==========================================================
# TOP ANOMALIES
# ==========================================================

print(
    "\nTop 10 V2 anomalies:"
)

top = (
    results
    .sort_values(
        "anomaly_score_v2",
        ascending=False
    )
    .head(10)
)

columns = [
    "ts",
    "it_power_kw",
    "pue",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "anomaly_score_v2"
]

print(
    top[columns].to_string(
        index=False
    )
)


# ==========================================================
# SAVE
# ==========================================================

results.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_PATH}"
)

print(
    "\nV2 training completed."
)