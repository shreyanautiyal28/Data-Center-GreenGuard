import pandas as pd


def validate_operational_plausibility(df):
    """
    Flags observations that are statistically unusual but may be
    unreliable for sustainability interpretation.

    This does NOT delete anomalies.
    It separates operationally plausible anomalies from
    low-load / telemetry edge cases.
    """

    df = df.copy()

    quality_flags = []
    quality_reasons = []

    for _, row in df.iterrows():

        flags = []
        reasons = []

        it_power = row.get("it_power_kw")
        pue = row.get("pue")
        cooling_ratio = row.get("cooling_to_it_ratio")
        hvac_ratio = row.get("hvac_to_it_ratio")
        pump_ratio = row.get("pump_to_it_ratio")

        if pd.notna(it_power) and it_power <= 10:
            flags.append("low_it_load")
            reasons.append(
                f"IT power is very low ({it_power:.2f} kW)"
            )

        if pd.notna(pue) and pue > 3:
            flags.append("extreme_pue")
            reasons.append(
                f"PUE is extremely high ({pue:.2f})"
            )

        if pd.notna(cooling_ratio) and cooling_ratio > 1:
            flags.append("extreme_cooling_ratio")
            reasons.append(
                f"Cooling-to-IT ratio is extreme ({cooling_ratio:.2f})"
            )

        if pd.notna(hvac_ratio) and hvac_ratio > 1:
            flags.append("extreme_hvac_ratio")
            reasons.append(
                f"HVAC-to-IT ratio is extreme ({hvac_ratio:.2f})"
            )

        if pd.notna(pump_ratio) and pump_ratio > 1:
            flags.append("extreme_pump_ratio")
            reasons.append(
                f"Pump-to-IT ratio is extreme ({pump_ratio:.2f})"
            )

        if flags:
            quality_flags.append("Review-required")
        else:
            quality_flags.append("Operationally-plausible")

        quality_reasons.append("; ".join(reasons))

    df["data_quality_status"] = quality_flags
    df["data_quality_reason"] = quality_reasons

    return df