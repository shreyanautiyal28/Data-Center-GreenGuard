import pandas as pd

DATA_PATH = "data/raw/public/esif_pue.parquet"

print("Loading NLR ESIF dataset...")

df = pd.read_parquet(DATA_PATH)

print("\n===================================")
print("NLR ESIF DATASET INSPECTION")
print("===================================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())