import pandas as pd


def classify_root_cause(df):
    """
    Evidence-based heuristic root-cause ranking.

    IMPORTANT:
    This identifies likely contributing operational factors.
    It does NOT establish physical causality.
    """

    df = df.copy()

    primary_causes = []
    contributing_factors = []
    evidence_list = []

    for _, row in df.iterrows():

        pue = row.get("pue")
        cooling_ratio = row.get("cooling_to_it_ratio")
        hvac_ratio = row.get("hvac_to_it_ratio")
        pump_ratio = row.get("pump_to_it_ratio")
        it_power = row.get("it_power_kw")

        factors = []
        evidence = []

        # --------------------------------------------------
        # HVAC
        # --------------------------------------------------

        if not pd.isna(hvac_ratio) and hvac_ratio >= 0.05:
            factors.append(("HVAC-related", hvac_ratio))
            evidence.append(
                f"HVAC-to-IT ratio elevated ({hvac_ratio:.3f})"
            )

        # --------------------------------------------------
        # Pump
        # --------------------------------------------------

        if not pd.isna(pump_ratio) and pump_ratio >= 0.03:
            factors.append(("Pump-related", pump_ratio))
            evidence.append(
                f"Pump-to-IT ratio elevated ({pump_ratio:.3f})"
            )

        # --------------------------------------------------
        # Cooling
        # --------------------------------------------------

        if not pd.isna(cooling_ratio) and cooling_ratio >= 0.20:
            factors.append(("Cooling-related", cooling_ratio))
            evidence.append(
                f"Cooling-to-IT ratio elevated ({cooling_ratio:.3f})"
            )

        # --------------------------------------------------
        # Very low IT load
        # --------------------------------------------------

        if not pd.isna(it_power) and it_power <= 10:
            factors.append(("IT-load-related", it_power))
            evidence.append(
                f"IT power unusually low ({it_power:.2f} kW)"
            )

        # --------------------------------------------------
        # PUE evidence
        # --------------------------------------------------

        if not pd.isna(pue) and pue >= 1.10:
            evidence.append(
                f"PUE elevated ({pue:.3f})"
            )

        # --------------------------------------------------
        # Determine primary cause
        # --------------------------------------------------

        if not factors:

            primary = "Multi-factor"
            evidence.append(
                "No individual subsystem crossed the current heuristic thresholds."
            )

        else:

            # Use the strongest normalized signal as the primary factor.
            # These values are heuristic rankings, not causal probabilities.
            primary = max(
                factors,
                key=lambda x: x[1]
            )[0]

        # --------------------------------------------------
        # Contributing factors
        # --------------------------------------------------

        if factors:

            contributing = ", ".join(
                factor[0]
                for factor in factors
            )

        else:

            contributing = "No dominant subsystem identified"

        primary_causes.append(primary)
        contributing_factors.append(contributing)
        evidence_list.append("; ".join(evidence))

    df["root_cause"] = primary_causes
    df["contributing_factors"] = contributing_factors
    df["root_cause_evidence"] = evidence_list

    return df