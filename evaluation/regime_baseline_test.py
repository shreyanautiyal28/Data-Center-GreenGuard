import pandas as pd
import numpy as np


FEATURE_PATH = "data/processed/esif_pue_features.parquet"


print("=" * 70)
print("GREENGUARD REGIME-AWARE EXPECTED-BEHAVIOR BASELINE")
print("=" * 70)

# ============================================================
# 1. Load
# ============================================================

print("\nLoading dataset...")

df = pd.read_parquet(FEATURE_PATH)

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

df = df.sort_values("ts").reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")

# ============================================================
# 2. Operational filtering
# ============================================================

required = [
    "ts",
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue"
]

df = df.dropna(subset=required).copy()

df = df[
    (df["it_power_kw"] > 100)
    & (df["pue"] >= 0.95)
    & (df["pue"] <= 1.15)
    & (df["cooling_kw"] >= 0)
    & (df["hvac_kw"] >= 0)
    & (df["pump_kw"] >= 0)
].copy()

# ============================================================
# 3. Temporal features
# ============================================================

df["hour"] = df["ts"].dt.hour
df["day_of_week"] = df["ts"].dt.dayofweek
df["month"] = df["ts"].dt.month

# ============================================================
# 4. IT-load regime
# ============================================================

bins = [
    100,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    np.inf
]

labels = [
    "100-500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    "2000-2500",
    "2500-3000",
    "3000-3500",
    "3500-4000",
    "4000+"
]

df["it_regime"] = pd.cut(
    df["it_power_kw"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# ============================================================
# 5. Chronological split
# ============================================================

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print(
    f"\nTraining rows: {len(train):,}"
)

print(
    f"Testing rows : {len(test):,}"
)

print(
    f"\nTraining period:\n"
    f"{train['ts'].min()} -> {train['ts'].max()}"
)

print(
    f"\nTesting period:\n"
    f"{test['ts'].min()} -> {test['ts'].max()}"
)

# ============================================================
# 6. Build regime statistics from TRAIN ONLY
# ============================================================

TARGETS = [
    "cooling_kw",
    "hvac_kw",
    "pump_kw"
]

print("\nBuilding training-period regime baselines...")

baseline = (
    train
    .groupby("it_regime", observed=True)[TARGETS]
    .median()
)

print("\nTraining regime baselines:")
print(baseline.to_string())

# ============================================================
# 7. Apply baseline to test
# ============================================================

for target in TARGETS:

    expected_column = f"expected_{target}"

    test[expected_column] = (
        test["it_regime"]
        .map(baseline[target])
    )

# ============================================================
# 8. Determine coverage
# ============================================================

print("\nBaseline coverage:")

for target in TARGETS:

    column = f"expected_{target}"

    coverage = (
        test[column].notna().mean() * 100
    )

    print(
        f"{target:<15}: "
        f"{coverage:.2f}%"
    )

# ============================================================
# 9. Calculate deviations
# ============================================================

for subsystem in [
    "cooling",
    "hvac",
    "pump"
]:

    actual = f"{subsystem}_kw"
    expected = f"expected_{subsystem}_kw"

    test[f"{subsystem}_deviation_kw"] = (
        test[actual] - test[expected]
    )

    test[f"{subsystem}_deviation_ratio"] = (
        test[f"{subsystem}_deviation_kw"]
        / test[expected].clip(lower=1.0)
    )

# ============================================================
# 10. Statistics
# ============================================================

print("\nDeviation statistics:")
print("-" * 70)

for subsystem in [
    "cooling",
    "hvac",
    "pump"
]:

    column = f"{subsystem}_deviation_ratio"

    valid = test[column].dropna()

    print(f"\n{subsystem.upper()}")

    print(
        f"Mean   : {valid.mean():.4f}"
    )

    print(
        f"Median : {valid.median():.4f}"
    )

    print(
        f"Std    : {valid.std():.4f}"
    )

    print(
        f"95th % : {valid.quantile(0.95):.4f}"
    )

    print(
        f"99th % : {valid.quantile(0.99):.4f}"
    )

    print(
        f"Max    : {valid.max():.4f}"
    )

    print(
        f"> +100%: "
        f"{(valid > 1).mean() * 100:.2f}%"
    )

# ============================================================
# 11. Test regime distribution
# ============================================================

print("\nTest IT-load regime distribution:")
print("-" * 70)

print(
    test["it_regime"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# 12. Save
# ============================================================

output_columns = [
    "ts",
    "it_power_kw",
    "it_regime",

    "cooling_kw",
    "expected_cooling_kw",
    "cooling_deviation_kw",
    "cooling_deviation_ratio",

    "hvac_kw",
    "expected_hvac_kw",
    "hvac_deviation_kw",
    "hvac_deviation_ratio",

    "pump_kw",
    "expected_pump_kw",
    "pump_deviation_kw",
    "pump_deviation_ratio",

    "pue"
]

output_path = (
    "data/processed/"
    "regime_baseline_validation.parquet"
)

test[output_columns].to_parquet(
    output_path,
    index=False
)

print(
    f"\nSaved:\n{output_path}"
)

print("\nRegime-aware baseline validation completed.")