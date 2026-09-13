import pandas as pd
import numpy as np


FEATURE_PATH = "data/processed/esif_pue_features.parquet"

print("=" * 70)
print("GREENGUARD TEMPORAL DISTRIBUTION-SHIFT DIAGNOSTIC")
print("=" * 70)

# ============================================================
# 1. Load
# ============================================================

df = pd.read_parquet(FEATURE_PATH)

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

df = df.sort_values("ts").reset_index(drop=True)

# ============================================================
# 2. Same operational filtering used by baseline
# ============================================================

required = [
    "ts",
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue"
]

df = df.dropna(subset=required)

df = df[
    (df["it_power_kw"] > 100)
    & (df["pue"] >= 0.95)
    & (df["pue"] <= 1.15)
    & (df["cooling_kw"] >= 0)
    & (df["hvac_kw"] >= 0)
    & (df["pump_kw"] >= 0)
].copy()

# ============================================================
# 3. Chronological split
# ============================================================

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print(f"\nTraining rows: {len(train):,}")
print(f"Testing rows : {len(test):,}")

print(
    f"\nTraining period:\n"
    f"{train['ts'].min()} -> {train['ts'].max()}"
)

print(
    f"\nTesting period:\n"
    f"{test['ts'].min()} -> {test['ts'].max()}"
)

# ============================================================
# 4. Compare distributions
# ============================================================

variables = [
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue"
]

print("\n" + "=" * 70)
print("TRAIN VS TEST DISTRIBUTIONS")
print("=" * 70)

for column in variables:

    print(f"\n{column}")

    print(
        f"  TRAIN median : {train[column].median():.4f}"
    )

    print(
        f"  TEST median  : {test[column].median():.4f}"
    )

    print(
        f"  TRAIN mean   : {train[column].mean():.4f}"
    )

    print(
        f"  TEST mean    : {test[column].mean():.4f}"
    )

    print(
        f"  TRAIN 95%    : {train[column].quantile(0.95):.4f}"
    )

    print(
        f"  TEST 95%     : {test[column].quantile(0.95):.4f}"
    )

# ============================================================
# 5. Compare subsystem ratios
# ============================================================

for frame in [train, test]:

    frame["cooling_to_it"] = (
        frame["cooling_kw"]
        / frame["it_power_kw"]
    )

    frame["hvac_to_it"] = (
        frame["hvac_kw"]
        / frame["it_power_kw"]
    )

    frame["pump_to_it"] = (
        frame["pump_kw"]
        / frame["it_power_kw"]
    )

print("\n" + "=" * 70)
print("SUBSYSTEM / IT RATIOS")
print("=" * 70)

for column in [
    "cooling_to_it",
    "hvac_to_it",
    "pump_to_it"
]:

    print(f"\n{column}")

    print(
        f"  TRAIN median : {train[column].median():.6f}"
    )

    print(
        f"  TEST median  : {test[column].median():.6f}"
    )

    print(
        f"  TRAIN 95%    : {train[column].quantile(0.95):.6f}"
    )

    print(
        f"  TEST 95%     : {test[column].quantile(0.95):.6f}"
    )

# ============================================================
# 6. HVAC by IT-load bands
# ============================================================

print("\n" + "=" * 70)
print("HVAC BEHAVIOR BY IT-LOAD BAND")
print("=" * 70)

bins = [100, 250, 500, 1000, 2000, 4000, np.inf]
labels = [
    "100-250",
    "250-500",
    "500-1000",
    "1000-2000",
    "2000-4000",
    "4000+"
]

for frame in [train, test]:

    frame["it_band"] = pd.cut(
        frame["it_power_kw"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

for band in labels:

    train_band = train[
        train["it_band"] == band
    ]

    test_band = test[
        test["it_band"] == band
    ]

    if len(train_band) == 0 or len(test_band) == 0:
        continue

    print(f"\nIT band: {band} kW")

    print(
        f"  TRAIN rows : {len(train_band):,}"
    )

    print(
        f"  TEST rows  : {len(test_band):,}"
    )

    print(
        f"  TRAIN HVAC median : "
        f"{train_band['hvac_kw'].median():.3f}"
    )

    print(
        f"  TEST HVAC median  : "
        f"{test_band['hvac_kw'].median():.3f}"
    )

    print(
        f"  TRAIN HVAC/IT     : "
        f"{train_band['hvac_to_it'].median():.5f}"
    )

    print(
        f"  TEST HVAC/IT      : "
        f"{test_band['hvac_to_it'].median():.5f}"
    )

print("\nDiagnostic completed.")