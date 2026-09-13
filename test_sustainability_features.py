import pandas as pd

from sustainability.features import (
    create_sustainability_features
)


INPUT_PATH = "data/processed/esif_pue_clean.parquet"
OUTPUT_PATH = "data/processed/esif_pue_features.parquet"


print("Loading cleaned NLR dataset...")

df = pd.read_parquet(INPUT_PATH)

print(f"Rows loaded: {len(df):,}")


print("\nCreating sustainability features...")

df = create_sustainability_features(df)


print("\n========================================")
print("SUSTAINABILITY FEATURES")
print("========================================")


feature_columns = [
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "plug_and_light_kw",
    "pue",
    "pue_proxy",
    "cooling_to_it_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "non_it_to_it_ratio",
    "ere",
]


print("\nFeature summary:")

print(
    df[feature_columns].describe()
)


print("\nSaving feature dataset...")

df.to_parquet(
    OUTPUT_PATH,
    index=False
)


print("\nFeature dataset saved:")
print(OUTPUT_PATH)

print("\nFinal columns:")

for column in df.columns:
    print(" -", column)

print("\nFeature engineering completed successfully.")