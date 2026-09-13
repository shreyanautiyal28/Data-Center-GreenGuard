import pandas as pd


BASELINE_PUE = 1.033


def calculate_severity(df):
    """
    Calculate a transparent sustainability priority score.

    IMPORTANT:
    This is a prioritization heuristic, not a measured probability
    of failure or a verified financial/energy-loss calculation.
    """

    df = df.copy()

    scores = []
    levels = []

    for _, row in df.iterrows():

        score = 0.0

        anomaly_score = row.get("anomaly_score")
        pue = row.get("pue")
        excess_power = row.get("estimated_excess_power_kw")

        # --------------------------------------------------
        # 1. Anomaly strength
        # --------------------------------------------------

        if pd.notna(anomaly_score):
            # Lower Isolation Forest score = more anomalous.
            anomaly_component = max(
                0,
                min(
                    30,
                    (0.20 - anomaly_score) / 0.20 * 30
                )
            )

            score += anomaly_component

        # --------------------------------------------------
        # 2. PUE deviation
        # --------------------------------------------------

        if pd.notna(pue):

            pue_excess = max(
                0,
                pue - BASELINE_PUE
            )

            pue_component = min(
                30,
                pue_excess / 0.50 * 30
            )

            score += pue_component

        # --------------------------------------------------
        # 3. Estimated excess power
        # --------------------------------------------------

        if pd.notna(excess_power):

            power_component = min(
                25,
                excess_power / 500 * 25
            )

            score += power_component

        # --------------------------------------------------
        # 4. Operational plausibility
        # --------------------------------------------------

        quality_status = row.get(
            "data_quality_status",
            "Operationally-plausible"
        )

        if quality_status != "Operationally-plausible":

            # Reduce priority because the observation requires
            # telemetry/workload validation first.
            score *= 0.50

        score = min(
            100,
            max(0, score)
        )

        # --------------------------------------------------
        # Severity classification
        # --------------------------------------------------

        if score >= 75:
            level = "Critical"

        elif score >= 50:
            level = "High"

        elif score >= 25:
            level = "Medium"

        else:
            level = "Low"

        scores.append(round(score, 2))
        levels.append(level)

    df["sustainability_priority_score"] = scores
    df["severity"] = levels

    return df