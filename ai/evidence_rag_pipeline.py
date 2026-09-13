import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from rag.retriever import SustainabilityRetriever
from ai.grounded_explainer import build_grounded_explanation

INPUT = "data/processed/greenguard_final_anomalies.parquet"

def build_evidence_query(row):
    """
    Convert quantitative anomaly evidence into a
    subsystem-aware sustainability RAG query.

    The likely_subsystem is treated as a hypothesis
    derived from quantitative evidence, NOT as confirmed
    root cause.
    """

    signals = []

    # -----------------------------------------------------
    # PRIMARY SUBSYSTEM CONTEXT
    # -----------------------------------------------------

    subsystem = row["likely_subsystem"]

    subsystem_context = (
        f"Likely contributing subsystem: {subsystem}. "
        "Treat this as an evidence-based investigation hypothesis, "
        "not a confirmed root cause."
    )

    # -----------------------------------------------------
    # PUE
    # -----------------------------------------------------

    if row["pue_deviation"] > 0:
        signals.append(
            f"elevated PUE ({row['pue']:.3f}, "
            f"deviation {row['pue_deviation']:.3f})"
        )

    # -----------------------------------------------------
    # HVAC
    # -----------------------------------------------------

    if row["hvac_deviation_ratio"] > 0.10:
        signals.append(
            f"elevated HVAC deviation "
            f"({row['hvac_deviation_ratio']:.2f})"
        )

    # -----------------------------------------------------
    # Pump
    # -----------------------------------------------------

    if row["pump_deviation_ratio"] > 0.10:
        signals.append(
            f"elevated pump deviation "
            f"({row['pump_deviation_ratio']:.2f})"
        )

    # -----------------------------------------------------
    # Cooling
    # -----------------------------------------------------

    if row["cooling_deviation_ratio"] > 0.10:
        signals.append(
            f"elevated cooling deviation "
            f"({row['cooling_deviation_ratio']:.2f})"
        )

    # -----------------------------------------------------
    # SUBSYSTEM RATIOS
    # -----------------------------------------------------

    if row["hvac_to_it_ratio"] > 0.05:
        signals.append(
            f"high HVAC-to-IT ratio "
            f"({row['hvac_to_it_ratio']:.3f})"
        )

    if row["pump_to_it_ratio"] > 0.03:
        signals.append(
            f"elevated pump-to-IT ratio "
            f"({row['pump_to_it_ratio']:.3f})"
        )

    if row["cooling_to_it_ratio"] > 0.20:
        signals.append(
            f"elevated cooling-to-IT ratio "
            f"({row['cooling_to_it_ratio']:.3f})"
        )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if not signals:
        signals.append(
            "statistically unusual data-center operation"
        )

    # -----------------------------------------------------
    # SUBSYSTEM-SPECIFIC RETRIEVAL INSTRUCTION
    # -----------------------------------------------------

    if subsystem == "HVAC":
        retrieval_focus = (
            "Prioritize HVAC efficiency, HVAC operating conditions, "
            "environmental conditions, schedules, and control settings."
        )

    elif subsystem == "Pump":
        retrieval_focus = (
            "Prioritize pump efficiency, flow, pressure, operating "
            "points, and pump control conditions."
        )

    elif subsystem == "Cooling":
        retrieval_focus = (
            "Prioritize cooling efficiency, cooling equipment, "
            "operating conditions, and cooling control settings."
        )

    else:
        retrieval_focus = (
            "Consider facility-level PUE efficiency and correlated "
            "subsystem conditions."
        )

    query = (
        "Data-center sustainability anomaly. "
        + subsystem_context
        + " Observed evidence: "
        + ", ".join(signals)
        + ". "
        + retrieval_focus
        + " Retrieve relevant sustainability knowledge, "
          "possible contributing operational factors, "
          "and safe investigation areas."
    )

    return query
def format_grounded_explanation(row, retrieved):
    """
    Produce a deterministic grounded explanation.

    No LLM is used here yet.
    This stage deliberately keeps the explanation
    traceable to observed evidence + retrieved knowledge.
    """

    evidence_lines = [
        f"Observed IT load: {row['it_power_kw']:.2f} kW",
        f"Observed PUE: {row['pue']:.4f}",
        f"Potential excess power: "
        f"{row['estimated_excess_power_kw']:.2f} kW",
        f"Likely contributing subsystem: "
        f"{row['likely_subsystem']}",
        f"Impact priority: "
        f"{row['impact_priority']}",
    ]

    knowledge_lines = []

    for item in retrieved:

        knowledge_lines.append({
            "topic": item["topic"],
            "similarity": round(item["score"], 4),
            "condition": item["condition"],
            "explanation": item["explanation"],
            "recommended_action": item["recommended_action"],
            "sustainability_area": item["sustainability_area"],
        })

    return {
        "observed_evidence": evidence_lines,
        "retrieved_knowledge": knowledge_lines,
    }


def run_pipeline(top_n=10, top_k=3):

    print("=" * 70)
    print("GREENGUARD EVIDENCE → RAG PIPELINE")
    print("=" * 70)

    print("\nLoading final anomaly dataset...")

    df = pd.read_parquet(INPUT)

    print(f"Anomalies available: {len(df):,}")

    retriever = SustainabilityRetriever()

    results = []

    for _, row in df.head(top_n).iterrows():

        query = build_evidence_query(row)

        retrieved = retriever.retrieve(
    		query,
    		top_k=top_k,
    		subsystem=row["likely_subsystem"]
	)	

        grounded = build_grounded_explanation(
    		row,
    		retrieved
	)


        results.append({
            "ts": row["ts"],
            "impact_rank": row["impact_rank"],
            "impact_priority": row["impact_priority"],
            "likely_subsystem": row["likely_subsystem"],
            "pue": row["pue"],
            "it_power_kw": row["it_power_kw"],
            "estimated_excess_power_kw":
                row["estimated_excess_power_kw"],
            "evidence_query": query,
            "grounded_result": grounded,
        })

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 70)
        print(f"ANOMALY #{i}")
        print("=" * 70)

        print(f"\nTimestamp: {result['ts']}")
        print(f"Impact rank: {result['impact_rank']}")
        print(f"Priority: {result['impact_priority']}")
        print(
            f"Likely subsystem: "
            f"{result['likely_subsystem']}"
        )

        print(
            f"\nPUE: {result['pue']:.4f}"
        )

        print(
            f"IT load: "
            f"{result['it_power_kw']:.2f} kW"
        )

        print(
            f"Potential excess power: "
            f"{result['estimated_excess_power_kw']:.2f} kW"
        )

        print("\nEvidence-aware query:")
        print(result["evidence_query"])
        print("\nGrounded explanation:")
        print(result["grounded_result"]["summary"])

        print("\nSafe investigation actions:")

        for action in result["grounded_result"]["investigation_actions"]:
                print(f"- {action}")

        print("\nCaveats:")

        for caveat in result["grounded_result"]["caveats"]:
            print(f"- {caveat}")
        print("\nRetrieved sustainability knowledge:")

        for j, item in enumerate(
            result["grounded_result"]["retrieved_knowledge"],
            start=1
        ):

            print("\n" + "-" * 50)
            print(f"Result {j}")
            print("-" * 50)

            print(
                f"Similarity: "
                f"{item['similarity']}"
            )

            print(
                f"Topic: "
                f"{item['topic']}"
            )

            print(
                f"Condition: "
                f"{item['condition']}"
            )

            print(
                f"Explanation: "
                f"{item['explanation']}"
            )

            print(
                f"Recommended investigation: "
                f"{item['recommended_action']}"
            )

    print("\n" + "=" * 70)
    print("EVIDENCE → RAG VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(
        top_n=10,
        top_k=3
    )