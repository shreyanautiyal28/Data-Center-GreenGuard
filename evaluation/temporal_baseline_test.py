import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_PATH = "data/processed/esif_pue_features.parquet"


print("=" * 70)
print("GREENGUARD TEMPORAL EXPECTED-BEHAVIOR VALIDATION")
print("=" * 70)

# ============================================================
# 1. Load data
# ============================================================

print("\nLoading dataset...")

df = pd.read_parquet(FEATURE_PATH)

df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

df = df.sort_values("ts").reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Time range: {df['ts'].min()} -> {df['ts'].max()}")

# ============================================================
# 2. Keep operationally usable observations
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

print(f"Valid operational rows: {len(df):,}")

# ============================================================
# 3. Add temporal context
# ============================================================

df["hour"] = df["ts"].dt.hour
df["day_of_week"] = df["ts"].dt.dayofweek
df["month"] = df["ts"].dt.month

# Cyclical representation
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# ============================================================
# 4. Chronological train/test split
# ============================================================

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("\nChronological split:")
print(f"Training rows: {len(train):,}")
print(f"Testing rows : {len(test):,}")

print(
    f"Training period: "
    f"{train['ts'].min()} -> {train['ts'].max()}"
)

print(
    f"Testing period : "
    f"{test['ts'].min()} -> {test['ts'].max()}"
)

# ============================================================
# 5. Training features
# ============================================================

MODEL_FEATURES = [
    "it_power_kw",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos"
]

TARGETS = [
    "cooling_kw",
    "hvac_kw",
    "pump_kw"
]

# ============================================================
# 6. Train models
# ============================================================

models = {}

print("\nTraining expected-behavior models...")

TRAIN_SAMPLE_SIZE = min(200_000, len(train))

train_sample = train.sample(
    n=TRAIN_SAMPLE_SIZE,
    random_state=42
).copy()

print(
    f"\nUsing {len(train_sample):,} rows "
    f"from the training period for model fitting."
)

models = {}

print("\nTraining expected-behavior models...")

for target in TARGETS:

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_sample[MODEL_FEATURES],
        train_sample[target]
    )

    models[target] = model

    print(f"  {target}: trained")

    models[target] = model

    print(f"  {target}: trained")

# ============================================================
# 7. Predict on unseen future period
# ============================================================

print("\nGenerating predictions on unseen test period...")

for target, model in models.items():

    expected_column = f"expected_{target}"

    test[expected_column] = model.predict(
        test[MODEL_FEATURES]
    )

# ============================================================
# 8. Calculate errors
# ============================================================

print("\nPrediction performance:")
print("-" * 70)

for target in TARGETS:

    expected = f"expected_{target}"

    mae = mean_absolute_error(
        test[target],
        test[expected]
    )

    print(
        f"{target:<15} "
        f"MAE = {mae:.4f} kW"
    )

# ============================================================
# 9. Calculate robust deviation features
# ============================================================

for subsystem in ["cooling", "hvac", "pump"]:

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
# 10. Clip extreme ratios for ML stability
# ============================================================

for subsystem in ["cooling", "hvac", "pump"]:

    column = f"{subsystem}_deviation_ratio"

    test[f"{subsystem}_deviation_ratio_clipped"] = (
        test[column].clip(-2, 5)
    )

# ============================================================
# 11. Display deviation statistics
# ============================================================

print("\nDeviation statistics on unseen test period:")
print("-" * 70)

for subsystem in ["cooling", "hvac", "pump"]:

    column = f"{subsystem}_deviation_ratio"

    stats = test[column].describe()

    print(f"\n{subsystem.upper()}")

    print(
        f"Mean   : {stats['mean']:.4f}\n"
        f"Median : {stats['50%']:.4f}\n"
        f"Std    : {stats['std']:.4f}\n"
        f"95th % : {test[column].quantile(0.95):.4f}\n"
        f"99th % : {test[column].quantile(0.99):.4f}\n"
        f"Max    : {stats['max']:.4f}"
    )

# ============================================================
# 12. Identify high positive deviations
# ============================================================

print("\nHigh-deviation observations:")
print("-" * 70)

for subsystem in ["cooling", "hvac", "pump"]:

    column = f"{subsystem}_deviation_ratio"

    high = test[test[column] > 1.0]

    print(
        f"{subsystem:<10}: "
        f"{len(high):,} observations "
        f"({len(high) / len(test) * 100:.2f}%) "
        f"above +100% deviation"
    )

# ============================================================
# 13. Save validation output
# ============================================================

output_columns = [
    "ts",
    "it_power_kw",
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
    "temporal_baseline_validation.parquet"
)

test[output_columns].to_parquet(
    output_path,
    index=False
)

print(
    f"\nSaved validation dataset to:\n"
    f"{output_path}"
)

print("\nTemporal baseline validation completed.")