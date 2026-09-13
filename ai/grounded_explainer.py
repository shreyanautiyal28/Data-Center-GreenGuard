import pandas as pd


def build_grounded_explanation(row, retrieved):
    """
    Build a deterministic, evidence-grounded explanation.

    This version does NOT use an LLM yet.
    Every statement is derived from:
    1. Quantitative anomaly evidence
    2. Sustainability impact analysis
    3. Retrieved sustainability knowledge

    The likely subsystem is treated as a hypothesis,
    not a confirmed root cause.
    """

    subsystem = row["likely_subsystem"]

    # --------------------------------------------------
    # OBSERVED EVIDENCE
    # --------------------------------------------------

    evidence = {
        "timestamp": str(row["ts"]),
        "it_load_kw": round(float(row["it_power_kw"]), 2),
        "pue": round(float(row["pue"]), 4),
        "pue_deviation": round(
            float(row["pue_deviation"]), 4
        ),
        "estimated_excess_power_kw": round(
            float(row["estimated_excess_power_kw"]), 2
        ),
        "hvac_deviation_ratio": round(
            float(row["hvac_deviation_ratio"]), 4
        ),
        "pump_deviation_ratio": round(
            float(row["pump_deviation_ratio"]), 4
        ),
        "cooling_deviation_ratio": round(
            float(row["cooling_deviation_ratio"]), 4
        ),
        "likely_subsystem": subsystem,
        "impact_priority": str(row["impact_priority"]),
        "evidence_class": str(row["evidence_class"]),
    }

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    statements = []

    statements.append(
        f"GreenGuard detected an operational anomaly at "
        f"{evidence['timestamp']}."
    )

    statements.append(
        f"The observed PUE was {evidence['pue']:.3f}, "
        f"which was {evidence['pue_deviation']:.3f} above "
        f"the normal-observation baseline."
    )

    if evidence["estimated_excess_power_kw"] > 0:
        statements.append(
            f"The event is associated with an estimated "
            f"{evidence['estimated_excess_power_kw']:.2f} kW "
            f"of potential excess facility power relative "
            f"to the baseline PUE."
        )
    else:
        statements.append(
            "The event did not produce a positive estimated "
            "excess-power value under the current baseline model."
        )

    # --------------------------------------------------
    # SUBSYSTEM-SPECIFIC EVIDENCE
    # --------------------------------------------------

    if subsystem == "HVAC":

        statements.append(
            "HVAC is the leading contributing subsystem "
            "according to the quantitative subsystem-evidence score."
        )

        if evidence["hvac_deviation_ratio"] > 0:
            statements.append(
                f"HVAC consumption was above its expected-behavior "
                f"baseline by a ratio of "
                f"{evidence['hvac_deviation_ratio']:.2f}."
            )

    elif subsystem == "Pump":

        statements.append(
            "Pumping is the leading contributing subsystem "
            "according to the quantitative subsystem-evidence score."
        )

        if evidence["pump_deviation_ratio"] > 0:
            statements.append(
                f"Pump consumption was above its expected-behavior "
                f"baseline by a ratio of "
                f"{evidence['pump_deviation_ratio']:.2f}."
            )

    elif subsystem == "Cooling":

        statements.append(
            "Cooling is the leading contributing subsystem "
            "according to the quantitative subsystem-evidence score."
        )

        if evidence["cooling_deviation_ratio"] > 0:
            statements.append(
                f"Cooling consumption was above its "
                f"expected-behavior baseline by a ratio of "
                f"{evidence['cooling_deviation_ratio']:.2f}."
            )

    else:

        statements.append(
            "No single subsystem crossed the current evidence "
            "threshold strongly enough to be designated dominant."
        )

    # --------------------------------------------------
    # RAG KNOWLEDGE
    # --------------------------------------------------

    knowledge = []

    for item in retrieved:

        knowledge.append({
            "topic": item["topic"],
            "similarity": round(
                float(item["score"]), 4
            ),
            "condition": item["condition"],
            "explanation": item["explanation"],
            "recommended_action": item["recommended_action"],
            "sustainability_area": item["sustainability_area"],
        })

    # --------------------------------------------------
    # SAFE INVESTIGATION ACTIONS
    # --------------------------------------------------

    investigation_actions = []

    # Prefer knowledge associated with the likely subsystem.
    for item in retrieved:

        topic = str(item["topic"]).lower()

        if subsystem.lower() in topic:
            investigation_actions.append(
                item["recommended_action"]
            )

    # Add relevant retrieved actions if subsystem match
    # was not found.
    if not investigation_actions:

        for item in retrieved[:2]:
            investigation_actions.append(
                item["recommended_action"]
            )

    # Remove duplicates while preserving order.
    investigation_actions = list(
        dict.fromkeys(investigation_actions)
    )

    # --------------------------------------------------
    # SAFETY / UNCERTAINTY
    # --------------------------------------------------

    caveats = [
        "The likely subsystem is an investigation hypothesis, "
        "not a confirmed equipment fault.",
        "Potential excess power is a modeled estimate relative "
        "to the selected baseline, not measured energy savings.",
        "Recommended actions are investigation steps and should "
        "be reviewed by an operator before any operational change."
    ]

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    return {
        "summary": " ".join(statements),
        "observed_evidence": evidence,
        "retrieved_knowledge": knowledge,
        "investigation_actions": investigation_actions,
        "caveats": caveats,
    }