import pandas as pd


DATA_PATH = "data/raw/public/esif_pue.parquet"

print("Loading real NLR dataset...")

df = pd.read_parquet(DATA_PATH)

print("\n========================================")
print("GREENGUARD REAL DATA PROFILE")
print("========================================")

# ---------------------------------------------------------
# Dataset size
# ---------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nMemory usage:")
print(
    round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    "MB"
)

# ---------------------------------------------------------
# Timestamp information
# ---------------------------------------------------------

df["ts"] = pd.to_datetime(df["ts"])

print("\nTimestamp range:")
print("Start:", df["ts"].min())
print("End:  ", df["ts"].max())

# ---------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate timestamps:")
print(df["ts"].duplicated().sum())

# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\nMissing values:")
missing = df.isna().sum()

missing_percent = (
    df.isna().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

print(missing_report)

# ---------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------

numeric_columns = [
    "cooling_kw",
    "hvac_kw",
    "it_power_kw",
    "plug_and_light_kw",
    "pue",
    "pump_kw",
    "ere"
]

print("\nNumeric summary:")

print(
    df[numeric_columns].describe()
)

# ---------------------------------------------------------
# Invalid / suspicious values
# ---------------------------------------------------------

print("\nNegative values:")

for column in numeric_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    print(
        f"{column}: {negative_count}"
    )

# ---------------------------------------------------------
# PUE sanity check
# ---------------------------------------------------------

print("\nPUE range:")

print(
    "Minimum:",
    df["pue"].min()
)

print(
    "Maximum:",
    df["pue"].max()
)

# ---------------------------------------------------------
# IT power range
# ---------------------------------------------------------

print("\nIT power range:")

print(
    "Minimum:",
    df["it_power_kw"].min()
)

print(
    "Maximum:",
    df["it_power_kw"].max()
)

print("\nProfiling completed.")