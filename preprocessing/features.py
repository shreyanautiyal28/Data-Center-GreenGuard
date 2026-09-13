import pandas as pd


def create_sustainability_features(df):
    """
    Create data-center sustainability and efficiency features.
    """

    df = df.copy()

    # ------------------------------------------------------
    # 1. Total facility power
    # ------------------------------------------------------
    df["facility_power_kw"] = (
        df["it_load_kw"] +
        df["cooling_load_kw"]
    )

    # ------------------------------------------------------
    # 2. Cooling-to-IT load ratio
    # ------------------------------------------------------
    df["cooling_it_ratio"] = (
        df["cooling_load_kw"] /
        df["it_load_kw"]
    )

    # ------------------------------------------------------
    # 3. PUE-like metric
    # ------------------------------------------------------
    df["pue"] = (
        df["facility_power_kw"] /
        df["it_load_kw"]
    )

    # ------------------------------------------------------
    # 4. Energy efficiency per utilization
    # ------------------------------------------------------
    df["energy_per_utilization"] = (
        df["total_energy_kwh"] /
        df["server_utilization_pct"].clip(lower=1)
    )

    # ------------------------------------------------------
    # 5. Water usage per kWh
    # ------------------------------------------------------
    df["water_per_kwh"] = (
        df["water_usage_l"] /
        df["total_energy_kwh"].clip(lower=1)
    )

    # ------------------------------------------------------
    # 6. Cooling efficiency indicator
    # ------------------------------------------------------
    df["cooling_efficiency"] = (
        df["it_load_kw"] /
        df["cooling_load_kw"].clip(lower=1)
    )

    # ------------------------------------------------------
    # 7. Temperature stress indicator
    # ------------------------------------------------------
    df["temperature_stress"] = (
        df["temperature_c"] >= 25
    ).astype(int)

    # ------------------------------------------------------
    # 8. High utilization indicator
    # ------------------------------------------------------
    df["high_utilization"] = (
        df["server_utilization_pct"] >= 85
    ).astype(int)

    # ------------------------------------------------------
    # 9. Cooling stress indicator
    # ------------------------------------------------------
    df["cooling_stress"] = (
        df["cooling_it_ratio"] >= 0.45
    ).astype(int)

    return df