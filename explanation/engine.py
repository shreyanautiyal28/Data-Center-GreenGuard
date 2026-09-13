import pandas as pd


def generate_explanation(row, retrieved_knowledge):
    """
    Combine anomaly evidence, root-cause analysis,
    and retrieved sustainability knowledge.

    This produces a structured decision-support explanation.
    It does not claim proven physical causality.
    """

    root_cause = row.get(
        "root_cause",
        "Unknown"
    )

    contributing = row.get(
        "contributing_factors",
        "Unknown"
    )

    pue = row.get("pue")
    it_power = row.get("it_power_kw")
    cooling_ratio = row.get("cooling_to_it_ratio")
    anomaly_score = row.get("anomaly_score")

    evidence = row.get(
        "root_cause_evidence",
        "No evidence available."
    )

    knowledge = []

    for item in retrieved_knowledge:
        knowledge.append({
            "topic": item["topic"],
            "component": item["component"],
            "explanation": item["explanation"],
            "recommended_action": item["recommended_action"],
            "similarity": item["score"]
        })

    return {
        "classification": root_cause,
        "contributing_factors": contributing,
        "anomaly_score": anomaly_score,
        "it_power_kw": it_power,
        "pue": pue,
        "cooling_to_it_ratio": cooling_ratio,
        "observed_evidence": evidence,
        "retrieved_knowledge": knowledge,
        "interpretation": (
            f"The observation was classified as {root_cause}. "
            f"Available telemetry indicates: {evidence}. "
            "These signals identify a likely area for investigation, "
            "but do not establish physical causality."
        )
    }