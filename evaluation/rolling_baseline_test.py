import pandas as pd
import numpy as np

FEATURE_PATH = "data/processed/esif_pue_features.parquet"

print("=" * 70)
print("GREENGUARD FAST WALK-FORWARD REGIME-AWARE BASELINE")
print("=" * 70)

# ==========================================================
# LOAD
# ==========================================================

print("\nLoading dataset...")

df = pd.read_parquet(FEATURE_PATH)

df["ts"] = pd.to_datetime(
    df["ts"],
    errors="coerce"
)

df = (
    df
    .sort_values("ts")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df):,}")


# ==========================================================
# OPERATIONAL FILTER
# ==========================================================

required = [
    "ts",
    "it_power_kw",
    "cooling_kw",
    "hvac_kw",
    "pump_kw",
    "pue"
]

df = df.dropna(
    subset=required
).copy()

df = df[
    (df["it_power_kw"] > 100)
    & (df["pue"] >= 0.95)
    & (df["pue"] <= 1.15)
    & (df["cooling_kw"] >= 0)
    & (df["hvac_kw"] >= 0)
    & (df["pump_kw"] >= 0)
].copy()

print(
    f"Operational rows: {len(df):,}"
)


# ==========================================================
# IT LOAD REGIMES
# ==========================================================

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


# ==========================================================
# TEMPORAL SPLIT
# ==========================================================

split_index = int(
    len(df) * 0.80
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()

print(
    f"\nTraining rows: {len(train):,}"
)

print(
    f"Testing rows : {len(test):,}"
)

print(
    f"\nTraining period:\n"
    f"{train['ts'].min()} -> "
    f"{train['ts'].max()}"
)

print(
    f"\nTesting period:\n"
    f"{test['ts'].min()} -> "
    f"{test['ts'].max()}"
)


# ==========================================================
# SETTINGS
# ==========================================================

WINDOW_DAYS = 180
MIN_HISTORY = 30
STEP = 100

SUBSYSTEMS = [
    "cooling",
    "hvac",
    "pump"
]

print(
    f"\nHistorical window: "
    f"{WINDOW_DAYS} days"
)

print(
    f"Minimum history: "
    f"{MIN_HISTORY}"
)

print(
    f"Evaluation step: "
    f"every {STEP}th test observation"
)


# ==========================================================
# PREPARE HISTORY
#
# IMPORTANT:
# We maintain separate numpy arrays for each IT regime.
#
# This avoids scanning millions of rows for every test point.
# ==========================================================

history = {}

for regime in labels:

    regime_train = train[
        train["it_regime"] == regime
    ].copy()

    history[regime] = {
        "ts": regime_train["ts"].values,

        "cooling": regime_train[
            "cooling_kw"
        ].values,

        "hvac": regime_train[
            "hvac_kw"
        ].values,

        "pump": regime_train[
            "pump_kw"
        ].values
    }


# ==========================================================
# WALK-FORWARD EVALUATION
# ==========================================================

evaluation_indices = np.arange(
    0,
    len(test),
    STEP
)

print(
    f"\nTest observations available: "
    f"{len(test):,}"
)

print(
    f"Test observations evaluated: "
    f"{len(evaluation_indices):,}"
)

print(
    "\nBuilding fast leakage-safe "
    "walk-forward baseline..."
)


results = []


for counter, test_position in enumerate(
    evaluation_indices
):

    row = test.iloc[
        test_position
    ]

    current_time = row["ts"]

    regime = row["it_regime"]

    window_start = (
        current_time
        - pd.Timedelta(
            days=WINDOW_DAYS
        )
    )


    # ------------------------------------------------------
    # Retrieve history for THIS regime only
    # ------------------------------------------------------

    regime_key = str(regime)

    if regime_key not in history:

        historical_ts = np.array([])

        cooling_history = np.array([])

        hvac_history = np.array([])

        pump_history = np.array([])

    else:

        h = history[regime_key]

        historical_ts = h["ts"]

        cooling_history = h["cooling"]

        hvac_history = h["hvac"]

        pump_history = h["pump"]


    # ------------------------------------------------------
    # Timestamp filtering using searchsorted
    #
    # Only observations:
    #
    # window_start <= timestamp < current_time
    #
    # are used.
    # ------------------------------------------------------

    left = np.searchsorted(
        historical_ts,
        np.datetime64(window_start),
        side="left"
    )

    right = np.searchsorted(
        historical_ts,
        np.datetime64(current_time),
        side="left"
    )


    history_count = (
        right - left
    )


    # ------------------------------------------------------
    # Baseline
    # ------------------------------------------------------

    expected_cooling = np.nan
    expected_hvac = np.nan
    expected_pump = np.nan


    if history_count >= MIN_HISTORY:

        expected_cooling = np.median(
            cooling_history[
                left:right
            ]
        )

        expected_hvac = np.median(
            hvac_history[
                left:right
            ]
        )

        expected_pump = np.median(
            pump_history[
                left:right
            ]
        )


    results.append({

        "ts":
            current_time,

        "it_power_kw":
            row["it_power_kw"],

        "it_regime":
            regime,

        "cooling_kw":
            row["cooling_kw"],

        "expected_cooling_kw":
            expected_cooling,

        "hvac_kw":
            row["hvac_kw"],

        "expected_hvac_kw":
            expected_hvac,

        "pump_kw":
            row["pump_kw"],

        "expected_pump_kw":
            expected_pump,

        "pue":
            row["pue"],

        "baseline_rows":
            history_count
    })


    # ------------------------------------------------------
    # AFTER prediction:
    #
    # Add CURRENT observation to history.
    #
    # This guarantees no future leakage.
    # ------------------------------------------------------

    if regime_key in history:

        history[regime_key]["ts"] = np.append(
            history[regime_key]["ts"],
            np.datetime64(current_time)
        )

        history[regime_key]["cooling"] = np.append(
            history[regime_key]["cooling"],
            row["cooling_kw"]
        )

        history[regime_key]["hvac"] = np.append(
            history[regime_key]["hvac"],
            row["hvac_kw"]
        )

        history[regime_key]["pump"] = np.append(
            history[regime_key]["pump"],
            row["pump_kw"]
        )


    if (
        counter > 0
        and counter % 500 == 0
    ):

        print(
            f"Processed "
            f"{counter:,} / "
            f"{len(evaluation_indices):,}"
        )


# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame(
    results
)


# ==========================================================
# DEVIATIONS
# ==========================================================

for subsystem in SUBSYSTEMS:

    actual = (
        f"{subsystem}_kw"
    )

    expected = (
        f"expected_{subsystem}_kw"
    )

    deviation_kw = (
        f"{subsystem}_deviation_kw"
    )

    deviation_ratio = (
        f"{subsystem}_deviation_ratio"
    )


    results[deviation_kw] = (
        results[actual]
        - results[expected]
    )


    results[deviation_ratio] = (
        results[deviation_kw]
        / results[expected].clip(
            lower=1.0
        )
    )


# ==========================================================
# VERIFY
# ==========================================================

print(
    "\nChecking generated baseline columns..."
)

for subsystem in SUBSYSTEMS:

    column = (
        f"expected_{subsystem}_kw"
    )

    if column not in results.columns:

        raise RuntimeError(
            f"Missing column: {column}"
        )

    print(
        f"OK: {column}"
    )


# ==========================================================
# COVERAGE
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "WALK-FORWARD BASELINE COVERAGE"
)

print(
    "=" * 70
)

for subsystem in SUBSYSTEMS:

    column = (
        f"expected_{subsystem}_kw"
    )

    coverage = (
        results[column]
        .notna()
        .mean()
        * 100
    )

    print(
        f"{subsystem:<15}: "
        f"{coverage:.2f}%"
    )


# ==========================================================
# DEVIATION STATISTICS
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "WALK-FORWARD DEVIATION"
)

print(
    "=" * 70
)

for subsystem in SUBSYSTEMS:

    column = (
        f"{subsystem}_deviation_ratio"
    )

    valid = (
        results[column]
        .dropna()
    )

    print(
        f"\n{subsystem.upper()}"
    )

    if len(valid) == 0:

        print(
            "No valid baseline observations."
        )

        continue

    print(
        f"Mean   : "
        f"{valid.mean():.4f}"
    )

    print(
        f"Median : "
        f"{valid.median():.4f}"
    )

    print(
        f"Std    : "
        f"{valid.std():.4f}"
    )

    print(
        f"95th % : "
        f"{valid.quantile(0.95):.4f}"
    )

    print(
        f"99th % : "
        f"{valid.quantile(0.99):.4f}"
    )

    print(
        f"Max    : "
        f"{valid.max():.4f}"
    )

    print(
        f"> +100%: "
        f"{(valid > 1).mean() * 100:.2f}%"
    )


# ==========================================================
# REGIME COVERAGE
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "BASELINE HISTORY BY IT REGIME"
)

print(
    "=" * 70
)

regime_stats = (
    results
    .groupby(
        "it_regime",
        observed=True
    )
    .agg(
        observations=(
            "ts",
            "count"
        ),

        baseline_rows_median=(
            "baseline_rows",
            "median"
        ),

        baseline_rows_min=(
            "baseline_rows",
            "min"
        ),

        baseline_rows_max=(
            "baseline_rows",
            "max"
        )
    )
)

print(
    regime_stats.to_string()
)


# ==========================================================
# SAVE
# ==========================================================

output_path = (
    "data/processed/"
    "rolling_baseline_validation.parquet"
)

results.to_parquet(
    output_path,
    index=False
)

print(
    f"\nSaved:\n"
    f"{output_path}"
)

print(
    "\nWalk-forward baseline "
    "validation completed."
)