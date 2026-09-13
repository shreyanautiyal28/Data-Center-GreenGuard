import pandas as pd
import numpy as np

INPUT = "data/processed/final_v2_impact_validation.parquet"
OUTPUT = "data/processed/greenguard_final_anomalies.parquet"

print("=" * 70)
print("GREENGUARD FINAL SUSTAINABILITY IMPACT DATASET")
print("=" * 70)

df = pd.read_parquet(INPUT)

# ---------------------------------------------------------
# 1. ANOMALIES ONLY
# ---------------------------------------------------------
df = df[df["is_anomaly_v2"] == 1].copy()

print(f"\nV2 anomalies: {len(df):,}")

# ---------------------------------------------------------
# 2. PUE IMPACT
# ---------------------------------------------------------
# Baseline = median PUE of normal observations.
# Reconstruct from the full validation dataset.
# ---------------------------------------------------------

full = pd.read_parquet(INPUT)

normal = full[
    full["is_anomaly_v2"] == 0
]

baseline_pue = normal["pue"].median()

df["pue_deviation"] = (
    df["pue"] - baseline_pue
)

# Positive PUE deviation only
df["positive_pue_deviation"] = (
    df["pue_deviation"].clip(lower=0)
)

# ---------------------------------------------------------
# 3. POTENTIAL EXCESS POWER
# ---------------------------------------------------------

df["estimated_excess_power_kw"] = (
    df["it_power_kw"] *
    df["positive_pue_deviation"]
)

# ---------------------------------------------------------
# 4. SUBSYSTEM EVIDENCE
# ---------------------------------------------------------

df["hvac_positive_deviation"] = (
    df["hvac_deviation_ratio"].clip(lower=0)
)

df["pump_positive_deviation"] = (
    df["pump_deviation_ratio"].clip(lower=0)
)

df["cooling_positive_deviation"] = (
    df["cooling_deviation_ratio"].clip(lower=0)
)

# ---------------------------------------------------------
# 5. SUBSYSTEM CONTRIBUTION SCORE
# ---------------------------------------------------------

df["hvac_signal_score"] = np.clip(
    df["hvac_positive_deviation"] * 20,
    0,
    100
)

df["pump_signal_score"] = np.clip(
    df["pump_positive_deviation"] * 20,
    0,
    100
)

df["cooling_signal_score"] = np.clip(
    df["cooling_positive_deviation"] * 20,
    0,
    100
)

# ---------------------------------------------------------
# 6. PUE IMPACT SCORE
# ---------------------------------------------------------

df["pue_impact_score"] = np.clip(
    df["positive_pue_deviation"] / 0.20 * 100,
    0,
    100
)

# ---------------------------------------------------------
# 7. POWER IMPACT SCORE
# ---------------------------------------------------------

power_95 = df["estimated_excess_power_kw"].quantile(0.95)

if power_95 > 0:
    df["power_impact_score"] = np.clip(
        df["estimated_excess_power_kw"] /
        power_95 * 100,
        0,
        100
    )
else:
    df["power_impact_score"] = 0

# ---------------------------------------------------------
# 8. ANOMALY STRENGTH SCORE
# ---------------------------------------------------------

score = df["anomaly_score_v2"]

score_min = score.min()
score_max = score.max()

if score_max > score_min:
    df["anomaly_strength_score"] = (
        (score - score_min) /
        (score_max - score_min) *
        100
    )
else:
    df["anomaly_strength_score"] = 50

# ---------------------------------------------------------
# 9. SUSTAINABILITY IMPACT SCORE
# ---------------------------------------------------------
#
# Sustainability is the primary objective.
#
# 35% PUE impact
# 30% potential excess power
# 20% subsystem evidence
# 15% anomaly strength
#
# This is a PRIORITIZATION score,
# not a probability of failure.
# ---------------------------------------------------------

df["subsystem_evidence_score"] = (
    df[
        [
            "hvac_signal_score",
            "pump_signal_score",
            "cooling_signal_score",
        ]
    ].max(axis=1)
)

df["sustainability_impact_score"] = (
    0.35 * df["pue_impact_score"]
    + 0.30 * df["power_impact_score"]
    + 0.20 * df["subsystem_evidence_score"]
    + 0.15 * df["anomaly_strength_score"]
)

# ---------------------------------------------------------
# 10. IMPACT PRIORITY
# ---------------------------------------------------------

df["impact_priority"] = pd.cut(
    df["sustainability_impact_score"],
    bins=[-np.inf, 25, 50, 75, np.inf],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical",
    ]
)

# ---------------------------------------------------------
# 11. LIKELY CONTRIBUTING SUBSYSTEM
# ---------------------------------------------------------

def identify_subsystem(row):

    signals = {
        "HVAC": row["hvac_signal_score"],
        "Pump": row["pump_signal_score"],
        "Cooling": row["cooling_signal_score"],
    }

    strongest = max(
        signals,
        key=signals.get
    )

    if signals[strongest] < 10:
        return "No dominant subsystem"

    return strongest


df["likely_subsystem"] = df.apply(
    identify_subsystem,
    axis=1
)

# ---------------------------------------------------------
# 12. EVIDENCE CLASSIFICATION
# ---------------------------------------------------------

def classify_evidence(row):

    if row["plausibility_status"] == "Review-required":
        return "Telemetry-review-required"

    if row["estimated_excess_power_kw"] > 0:
        return "Potential-sustainability-impact"

    return "Statistical-anomaly-only"


df["evidence_class"] = df.apply(
    classify_evidence,
    axis=1
)

# ---------------------------------------------------------
# 13. RANK
# ---------------------------------------------------------

df = df.sort_values(
    "sustainability_impact_score",
    ascending=False
).reset_index(drop=True)

df["impact_rank"] = (
    df.index + 1
)

# ---------------------------------------------------------
# 14. SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL IMPACT SUMMARY")
print("=" * 70)

print(
    f"\nBaseline PUE: {baseline_pue:.4f}"
)

print(
    "\nSustainability impact score:"
)

print(
    df["sustainability_impact_score"].describe()
)

print(
    "\nImpact priority:"
)

print(
    df["impact_priority"]
    .value_counts()
    .sort_index()
)

print(
    "\nLikely contributing subsystem:"
)

print(
    df["likely_subsystem"]
    .value_counts()
)

print(
    "\nEvidence classification:"
)

print(
    df["evidence_class"]
    .value_counts()
)

print(
    "\nTop 15 sustainability-priority anomalies:"
)

display_cols = [
    "impact_rank",
    "ts",
    "it_power_kw",
    "pue",
    "estimated_excess_power_kw",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "cooling_deviation_ratio",
    "likely_subsystem",
    "sustainability_impact_score",
    "impact_priority",
    "evidence_class",
]

print(
    df[display_cols]
    .head(15)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 15. SAVE
# ---------------------------------------------------------

df.to_parquet(
    OUTPUT,
    index=False
)

print(
    f"\nSaved final dataset:\n{OUTPUT}"
)

print("\n" + "=" * 70)
print("FINAL IMPACT DATASET COMPLETE")
print("=" * 70)