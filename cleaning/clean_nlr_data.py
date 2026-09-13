import pandas as pd


INPUT_PATH = "data/raw/public/esif_pue.parquet"
OUTPUT_PATH = "data/processed/esif_pue_clean.parquet"


print("Loading NLR ESIF dataset...")

df = pd.read_parquet(INPUT_PATH)

print(f"Original rows: {len(df):,}")


# ==========================================================
# 1. Remove unused metadata column
# ==========================================================

if "tags" in df.columns:
    df = df.drop(columns=["tags"])


# ==========================================================
# 2. Convert timestamp
# ==========================================================

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")


# ==========================================================
# 3. Sort chronologically
# ==========================================================

df = df.sort_values("ts").reset_index(drop=True)


# ==========================================================
# 4. Create data-quality flags
# ==========================================================

df["invalid_cooling"] = (
    df["cooling_kw"] < 0
)

df["invalid_hvac"] = (
    df["hvac_kw"] < 0
)

df["invalid_pump"] = (
    df["pump_kw"] < 0
)

df["invalid_ere"] = (
    df["ere"] < 0
)

df["invalid_pue"] = (
    df["pue"] < 1
)

df["extreme_pue"] = (
    df["pue"] > 2
)


# ==========================================================
# 5. Replace physically invalid sensor values with NaN
# ==========================================================

df.loc[
    df["cooling_kw"] < 0,
    "cooling_kw"
] = pd.NA

df.loc[
    df["hvac_kw"] < 0,
    "hvac_kw"
] = pd.NA

df.loc[
    df["pump_kw"] < 0,
    "pump_kw"
] = pd.NA

df.loc[
    df["ere"] < 0,
    "ere"
] = pd.NA

df.loc[
    df["pue"] < 1,
    "pue"
] = pd.NA


# ==========================================================
# 6. Create overall quality flag
# ==========================================================

quality_columns = [
    "invalid_cooling",
    "invalid_hvac",
    "invalid_pump",
    "invalid_ere",
    "invalid_pue",
]

df["has_quality_issue"] = (
    df[quality_columns].any(axis=1)
)


# ==========================================================
# 7. Create a modeling-ready subset
#
# We require the core energy variables.
# We DO NOT remove extreme positive values here.
# They may represent genuine anomalies.
# ==========================================================

core_columns = [
    "ts",
    "cooling_kw",
    "hvac_kw",
    "it_power_kw",
    "plug_and_light_kw",
    "pue",
    "pump_kw",
]

model_df = df.dropna(
    subset=core_columns
).copy()


# ==========================================================
# 8. Basic derived features
# ==========================================================

model_df["non_it_power_kw"] = (
    model_df["cooling_kw"]
    + model_df["hvac_kw"]
    + model_df["plug_and_light_kw"]
    + model_df["pump_kw"]
)

model_df["cooling_to_it_ratio"] = (
    model_df["cooling_kw"]
    / model_df["it_power_kw"].clip(lower=1)
)


# ==========================================================
# 9. Save cleaned dataset
# ==========================================================

model_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


print("\n========================================")
print("CLEANING COMPLETED")
print("========================================")

print(
    f"Original rows: {len(df):,}"
)

print(
    f"Modeling rows: {len(model_df):,}"
)

print(
    f"Removed rows: {len(df) - len(model_df):,}"
)

print(
    "\nRows with quality issues:",
    int(df["has_quality_issue"].sum())
)

print(
    "\nOutput:",
    OUTPUT_PATH
)

print("\nCleaned columns:")
print(model_df.columns.tolist())