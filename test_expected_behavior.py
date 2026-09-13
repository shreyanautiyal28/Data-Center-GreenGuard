import pandas as pd

from sustainability.expected_behavior import ExpectedBehaviorModel


FEATURE_PATH = (
    "data/processed/esif_pue_features.parquet"
)


print("=" * 60)
print("GREENGUARD EXPECTED-BEHAVIOR BASELINE TEST")
print("=" * 60)

print("\nLoading feature dataset...")

df = pd.read_parquet(FEATURE_PATH)

print(f"Rows loaded: {len(df):,}")

required_columns = [
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ------------------------------------------------------
# Use a manageable sample for the first test
# ------------------------------------------------------

df = df.sample(
    n=min(100_000, len(df)),
    random_state=42
).copy()

print(f"Rows used: {len(df):,}")

# ------------------------------------------------------
# Train expected-behavior model
# ------------------------------------------------------

model = ExpectedBehaviorModel()

print("\nTraining expected-behavior models...")

model.fit(df)

print("Training completed.")

# ------------------------------------------------------
# Generate predictions
# ------------------------------------------------------

result = model.transform(df)

# ------------------------------------------------------
# Display examples
# ------------------------------------------------------

print("\nExpected behavior examples:")
print(
    result[
        [
            "it_power_kw",
            "cooling_kw",
            "expected_cooling_kw",
            "cooling_deviation_ratio",
            "hvac_kw",
            "expected_hvac_kw",
            "hvac_deviation_ratio",
            "pump_kw",
            "expected_pump_kw",
            "pump_deviation_ratio"
        ]
    ].head(10).to_string(index=False)
)

# ------------------------------------------------------
# Summary
# ------------------------------------------------------

print("\nDeviation summary:")

for subsystem in [
    "cooling",
    "hvac",
    "pump"
]:

    column = f"{subsystem}_deviation_ratio"

    print(f"\n{subsystem.upper()}")

    print(
        result[column].describe()[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max"
            ]
        ]
    )

print("\nTest completed successfully.")