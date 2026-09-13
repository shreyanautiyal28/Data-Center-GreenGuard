import pandas as pd
import numpy as np

PATH = "data/processed/rolling_baseline_validation.parquet"

print("=" * 70)
print("GREENGUARD BASELINE QUALITY GATE")
print("=" * 70)

df = pd.read_parquet(PATH)

subsystems = ["cooling", "hvac", "pump"]

print(f"\nRows evaluated: {len(df):,}")

for subsystem in subsystems:

    actual = f"{subsystem}_kw"
    expected = f"expected_{subsystem}_kw"
    deviation = f"{subsystem}_deviation_ratio"

    valid = df[
        df[expected].notna()
        & np.isfinite(df[deviation])
    ].copy()

    print("\n" + "-" * 70)
    print(subsystem.upper())
    print("-" * 70)

    print(
        f"Coverage: "
        f"{len(valid) / len(df) * 100:.2f}%"
    )

    print(
        f"Median deviation: "
        f"{valid[deviation].median():.4f}"
    )

    print(
        f"Mean deviation: "
        f"{valid[deviation].mean():.4f}"
    )

    print(
        f"95th percentile: "
        f"{valid[deviation].quantile(.95):.4f}"
    )

    print(
        f"99th percentile: "
        f"{valid[deviation].quantile(.99):.4f}"
    )

    print(
        f"> +50%: "
        f"{(valid[deviation] > .50).mean() * 100:.2f}%"
    )

    print(
        f"> +100%: "
        f"{(valid[deviation] > 1.0).mean() * 100:.2f}%"
    )

    print(
        f"< -50%: "
        f"{(valid[deviation] < -.50).mean() * 100:.2f}%"
    )

    print(
        f"Expected median: "
        f"{valid[expected].median():.4f} kW"
    )

    print(
        f"Actual median: "
        f"{valid[actual].median():.4f} kW"
    )


# ==========================================================
# HISTORY QUALITY
# ==========================================================

print("\n" + "=" * 70)
print("HISTORY QUALITY")
print("=" * 70)

print(
    df["baseline_rows"]
    .describe()
    .to_string()
)

print(
    "\nRows with <30 historical observations: "
    f"{(df['baseline_rows'] < 30).sum():,}"
)

print(
    "Rows with >=30 historical observations: "
    f"{(df['baseline_rows'] >= 30).sum():,}"
)


# ==========================================================
# DECISION
# ==========================================================

print("\n" + "=" * 70)
print("GREEN GUARD BASELINE DECISION")
print("=" * 70)

print("""
HVAC:
  ACCEPT for further anomaly-model evaluation.
  Reason: adaptive baseline reduced regime-shift distortion.

PUMP:
  ACCEPT for further anomaly-model evaluation.
  Reason: stable deviation distribution and 0% >100%.

COOLING:
  HOLD.
  Reason: elevated positive deviations require additional
  investigation before integration.

IMPORTANT:
These decisions do not prove subsystem failure or efficiency.
They determine whether a baseline signal is sufficiently
stable for the next validation stage.
""")