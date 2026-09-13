import pandas as pd


INPUT_PATH = "data/processed/esif_pue_features.parquet"


print("Loading feature dataset...")

df = pd.read_parquet(INPUT_PATH)


# ==========================================================
# Reproduce the same ML sample
# ==========================================================

sample_df = df.sample(
    n=min(100_000, len(df)),
    random_state=42
).copy()


# ==========================================================
# Define sustainability risk indicators
# ==========================================================

sample_df["high_pue_flag"] = (
    sample_df["pue"] > 1.2
)

sample_df["high_cooling_ratio_flag"] = (
    sample_df["cooling_to_it_ratio"] > 0.02
)

sample_df["low_it_power_flag"] = (
    sample_df["it_power_kw"] < 10
)

sample_df["extreme_pue_flag"] = (
    sample_df["pue"] > 2
)


# ==========================================================
# Analyze sustainability signals
# ==========================================================

print("\n========================================")
print("ANOMALY SUSTAINABILITY ANALYSIS")
print("========================================")


print("\nOverall sample:")

print(
    f"Rows: {len(sample_df):,}"
)


print("\nPUE statistics:")

print(
    sample_df["pue"].describe()
)


print("\nHigh PUE observations:")

print(
    sample_df["high_pue_flag"].sum()
)


print("\nHigh cooling/IT ratio observations:")

print(
    sample_df["high_cooling_ratio_flag"].sum()
)


print("\nLow IT power observations:")

print(
    sample_df["low_it_power_flag"].sum()
)


print("\nExtreme PUE observations:")

print(
    sample_df["extreme_pue_flag"].sum()
)


# ==========================================================
# Relationship between signals
# ==========================================================

print("\n========================================")
print("SUSTAINABILITY SIGNAL SUMMARY")
print("========================================")


summary = pd.DataFrame({
    "signal": [
        "High PUE (>1.2)",
        "High cooling / IT ratio (>0.02)",
        "Low IT power (<10 kW)",
        "Extreme PUE (>2)"
    ],
    "count": [
        sample_df["high_pue_flag"].sum(),
        sample_df["high_cooling_ratio_flag"].sum(),
        sample_df["low_it_power_flag"].sum(),
        sample_df["extreme_pue_flag"].sum()
    ]
})

summary["percentage"] = (
    summary["count"]
    / len(sample_df)
    * 100
).round(2)


print(summary)


print("\nAnalysis completed.")