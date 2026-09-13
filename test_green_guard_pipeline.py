import pandas as pd

from root_cause.analyzer import classify_root_cause
from rag.retriever import SustainabilityRetriever
from explanation.engine import generate_explanation


ANOMALY_PATH = "data/processed/anomaly_results.parquet"


print("========================================")
print("DATA-CENTER GREENGUARD AI PIPELINE")
print("========================================")


# ---------------------------------------------------------
# 1. Load anomaly results
# ---------------------------------------------------------

print("\n[1/4] Loading anomaly results...")

df = pd.read_parquet(ANOMALY_PATH)

anomalies = df[
    df["anomaly_label"] == -1
].copy()

print(
    "Anomalies available:",
    len(anomalies)
)


# ---------------------------------------------------------
# 2. Root-cause analysis
# ---------------------------------------------------------

print("\n[2/4] Running root-cause analysis...")

anomalies = classify_root_cause(
    anomalies
)


# ---------------------------------------------------------
# 3. Initialize RAG
# ---------------------------------------------------------

print("\n[3/4] Loading RAG knowledge base...")

retriever = SustainabilityRetriever()


# ---------------------------------------------------------
# 4. Analyze one high-impact anomaly
# ---------------------------------------------------------

print("\n[4/4] Generating GreenGuard explanation...")


# Select anomaly with highest PUE
# among available anomaly observations.

target = anomalies.loc[
    anomalies["pue"].idxmax()
]


query = f"""
Data-center sustainability anomaly.

Root cause:
{target["root_cause"]}

Contributing factors:
{target["contributing_factors"]}

PUE:
{target["pue"]}

HVAC-to-IT ratio:
{target.get("hvac_to_it_ratio")}

Pump-to-IT ratio:
{target.get("pump_to_it_ratio")}

Cooling-to-IT ratio:
{target.get("cooling_to_it_ratio")}

Find relevant operational sustainability
knowledge and investigation recommendations.
"""


retrieved = retriever.retrieve(
    query,
    top_k=3
)


result = generate_explanation(
    target,
    retrieved
)


print("\n========================================")
print("GREENGUARD INTELLIGENCE RESULT")
print("========================================")

print("\nTimestamp:")
print(target["ts"])

print("\nRoot cause:")
print(result["classification"])

print("\nContributing factors:")
print(result["contributing_factors"])

print("\nIT power:")
print(result["it_power_kw"], "kW")

print("\nPUE:")
print(result["pue"])

print("\nCooling-to-IT ratio:")
print(result["cooling_to_it_ratio"])

print("\nAnomaly score:")
print(result["anomaly_score"])

print("\nObserved evidence:")
print(result["observed_evidence"])

print("\nInterpretation:")
print(result["interpretation"])


print("\n========================================")
print("RETRIEVED KNOWLEDGE")
print("========================================")


for i, item in enumerate(
    result["retrieved_knowledge"],
    1
):

    print(f"\n{i}. {item['topic']}")
    print(
        "Similarity:",
        round(item["similarity"], 4)
    )

    print(
        "Explanation:",
        item["explanation"]
    )

    print(
        "Recommended investigation:",
        item["recommended_action"]
    )


print("\nPipeline completed successfully.")