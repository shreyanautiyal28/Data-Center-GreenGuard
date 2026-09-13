import pandas as pd

from models.anomaly_detector import (
    train_anomaly_detector
)

INPUT_PATH = "data/processed/esif_pue_features.parquet"

print("Loading feature-enriched NLR dataset...")

df = pd.read_parquet(INPUT_PATH)

print(
    f"Total rows loaded: {len(df):,}"
)


# ==========================================================
# Sample data for initial model development
# ==========================================================

sample_size = min(
    100_000,
    len(df)
)

sample_df = df.sample(
    n=sample_size,
    random_state=42
).copy()

print(
    f"Rows used for initial ML training: {len(sample_df):,}"
)


# ==========================================================
# Train anomaly detector
# ==========================================================

model, scaler, results = train_anomaly_detector(
    sample_df
)


# ==========================================================
# Results
# ==========================================================

print("\n========================================")
print("GREENGUARD ANOMALY DETECTION")
print("========================================")

print(
    "\nTotal modeling rows:",
    len(results)
)

print(
    "\nNormal observations:",
    int(
        (results["anomaly_label"] == 1).sum()
    )
)

print(
    "Anomalies detected:",
    int(
        (results["anomaly_label"] == -1).sum()
    )
)

print(
    "\nAnomaly percentage:",
    round(
        (
            results["anomaly_label"] == -1
        ).mean() * 100,
        2
    ),
    "%"
)


print("\nMost anomalous observations:")

print(
    results[
        [
            "ts",
            "it_power_kw",
            "cooling_kw",
            "hvac_kw",
            "pue",
            "cooling_to_it_ratio",
            "anomaly_score",
            "anomaly_label",
        ]
    ]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(10)
)


print("\nML anomaly detection completed.")
results.to_parquet(
    "data/processed/anomaly_results.parquet",
    index=False
)

print(
    "\nAnomaly results saved:"
)

print(
    "data/processed/anomaly_results.parquet"
)