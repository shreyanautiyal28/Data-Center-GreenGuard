from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "esif_pue_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard_daily_summary.parquet"
)


# ==========================================================
# CONFIG
# ==========================================================

COLUMNS = [
    "ts",
    "pue",
    "it_power_kw",
    "cooling_kw",
]


# ==========================================================
# PROCESS PARQUET IN BATCHES
# ==========================================================

print("=" * 70)
print("GREENGUARD DASHBOARD SUMMARY BUILDER")
print("=" * 70)

print("\nLoading parquet in batches...")

parquet_file = pq.ParquetFile(
    INPUT_PATH
)

daily_parts = []

total_rows = 0

for batch_number, batch in enumerate(
    parquet_file.iter_batches(
        batch_size=100_000,
        columns=COLUMNS
    ),
    start=1
):

    chunk = batch.to_pandas()

    total_rows += len(chunk)

    chunk["ts"] = pd.to_datetime(
        chunk["ts"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["ts"]
    )

    chunk["date"] = (
        chunk["ts"]
        .dt.floor("D")
    )

    daily = (
        chunk
        .groupby("date")
        .agg(
            pue_median=("pue", "median"),
            it_power_mean=("it_power_kw", "mean"),
            cooling_power_mean=("cooling_kw", "mean"),
            records=("ts", "count"),
        )
        .reset_index()
    )

    daily_parts.append(
        daily
    )

    if batch_number % 10 == 0:

        print(
            f"Processed approximately "
            f"{total_rows:,} rows..."
        )


# ==========================================================
# COMBINE DAILY RESULTS
# ==========================================================

print("\nCombining daily summaries...")

daily_summary = pd.concat(
    daily_parts,
    ignore_index=True
)


# ==========================================================
# FINAL AGGREGATION
# ==========================================================

daily_summary = (
    daily_summary
    .groupby("date")
    .agg(
        pue_median=("pue_median", "median"),
        it_power_mean=("it_power_mean", "mean"),
        cooling_power_mean=("cooling_power_mean", "mean"),
        records=("records", "sum"),
    )
    .reset_index()
)


daily_summary = daily_summary.sort_values(
    "date"
)


# ==========================================================
# SAVE
# ==========================================================

daily_summary.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ==========================================================
# REPORT
# ==========================================================

print("\n" + "=" * 70)
print("DASHBOARD SUMMARY CREATED")
print("=" * 70)

print(
    f"Source rows processed : {total_rows:,}"
)

print(
    f"Daily records         : {len(daily_summary):,}"
)

print(
    f"Date range             : "
    f"{daily_summary['date'].min()} → "
    f"{daily_summary['date'].max()}"
)

print(
    f"Output                 : "
    f"{OUTPUT_PATH}"
)

print("=" * 70)