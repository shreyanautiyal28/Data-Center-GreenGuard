import pandas as pd


def calculate_sustainability_impact(df):
    """
    Estimate sustainability impact from operational inefficiency.

    IMPORTANT:
    These are modeled estimates, not measured savings.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Baseline PUE
    #
    # Use the median PUE of the available dataset as a
    # simple operational baseline.
    # ------------------------------------------------------

    baseline_pue = df["pue"].median()

    # ------------------------------------------------------
    # Estimated excess facility power
    #
    # Facility power = IT power * PUE
    #
    # Excess power above baseline:
    #
    # IT power * (observed PUE - baseline PUE)
    # ------------------------------------------------------

    df["baseline_pue"] = baseline_pue

    df["estimated_excess_power_kw"] = (
        df["it_power_kw"]
        * (
            df["pue"] - baseline_pue
        )
    )

    # Negative values mean the observation is better
    # than the baseline.
    df["estimated_excess_power_kw"] = (
        df["estimated_excess_power_kw"]
        .clip(lower=0)
    )

    # ------------------------------------------------------
    # Estimated excess energy for a one-hour equivalent
    #
    # Since observations are irregularly spaced, we do NOT
    # claim this is the actual energy consumed between rows.
    # This is a one-hour equivalent impact estimate.
    # ------------------------------------------------------

    df["estimated_excess_energy_kwh_per_hour"] = (
        df["estimated_excess_power_kw"]
    )

    return df