import pandas as pd


def create_sustainability_features(df):
    """
    Create sustainability-oriented features
    from the cleaned NLR ESIF data.
    """

    df = df.copy()

    # ======================================================
    # 1. Total non-IT facility power
    # ======================================================

    df["non_it_power_kw"] = (
        df["cooling_kw"]
        + df["hvac_kw"]
        + df["pump_kw"]
        + df["plug_and_light_kw"]
    )

    # ======================================================
    # 2. Thermal power burden
    # ======================================================

    df["thermal_power_kw"] = (
        df["cooling_kw"]
        + df["hvac_kw"]
    )

    # ======================================================
    # 3. Cooling-to-IT ratio
    # ======================================================

    df["cooling_to_it_ratio"] = (
        df["cooling_kw"]
        / df["it_power_kw"].clip(lower=1)
    )

    # ======================================================
    # 4. HVAC-to-IT ratio
    # ======================================================

    df["hvac_to_it_ratio"] = (
        df["hvac_kw"]
        / df["it_power_kw"].clip(lower=1)
    )

    # ======================================================
    # 5. Pump-to-IT ratio
    # ======================================================

    df["pump_to_it_ratio"] = (
        df["pump_kw"]
        / df["it_power_kw"].clip(lower=1)
    )

    # ======================================================
    # 6. Non-IT / IT ratio
    # ======================================================

    df["non_it_to_it_ratio"] = (
        df["non_it_power_kw"]
        / df["it_power_kw"].clip(lower=1)
    )

    # ======================================================
    # 7. Component-derived PUE proxy
    #
    # This is NOT claimed to be official PUE.
    # ======================================================

    df["pue_proxy"] = (
        1
        + (
            df["non_it_power_kw"]
            / df["it_power_kw"].clip(lower=1)
        )
    )

    # ======================================================
    # 8. Difference between official PUE and proxy
    # ======================================================

    df["pue_difference"] = (
        df["pue"] - df["pue_proxy"]
    )

    # ======================================================
    # 9. Energy reuse availability
    # ======================================================

    df["energy_reuse_available"] = (
        df["energy_reuse"].notna()
    ).astype(int)

    # ======================================================
    # 10. Energy reuse deviation
    # ======================================================

    df["ere_deviation"] = (
        df["ere"] - 1
    )

    # ======================================================
    # 11. High PUE indicator
    # ======================================================

    df["high_pue"] = (
        df["pue"] > 1.2
    ).astype(int)

    # ======================================================
    # 12. Return enhanced dataset
    # ======================================================

    return df