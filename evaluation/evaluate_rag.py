import sys
from pathlib import Path

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.retriever import SustainabilityRetriever


print("=" * 70)
print("GREENGUARD RAG RETRIEVAL EVALUATION")
print("=" * 70)


# -------------------------------------------------------------------
# Evaluation cases
# -------------------------------------------------------------------

TEST_CASES = [

    {
        "name": "HVAC anomaly",
        "query": """
        Data-center anomaly with elevated HVAC-to-IT ratio.
        HVAC consumption appears above expected behavior.
        Identify relevant HVAC sustainability knowledge.
        """,
        "subsystem": "HVAC",
        "expected_topic": "HVAC efficiency",
    },

    {
        "name": "Pump anomaly",
        "query": """
        Data-center anomaly with elevated pump-to-IT ratio.
        Pump consumption appears above expected behavior.
        Identify relevant pump sustainability knowledge.
        """,
        "subsystem": "Pump",
        "expected_topic": "Pump efficiency",
    },

    {
        "name": "Cooling anomaly",
        "query": """
        Data-center anomaly with elevated cooling-to-IT ratio.
        Cooling consumption appears above expected behavior.
        Identify relevant cooling sustainability knowledge.
        """,
        "subsystem": "Cooling",
        "expected_topic": "Cooling efficiency",
    },

    {
        "name": "PUE anomaly",
        "query": """
        Data-center observation has elevated PUE relative to
        the selected operational baseline.
        Identify relevant sustainability knowledge.
        """,
        "subsystem": None,
        "expected_topic": "PUE inefficiency",
    },

    {
        "name": "Low IT load",
        "query": """
        Data-center observation has unusually low IT load.
        Facility infrastructure consumption remains significant.
        Identify relevant sustainability knowledge.
        """,
        "subsystem": "IT",
        "expected_topic": "Low IT load",
    },

    {
        "name": "Multi-factor anomaly",
        "query": """
        Data-center anomaly involves multiple elevated HVAC,
        cooling, and pump subsystem signals simultaneously.
        The interacting facility systems may reflect a
        multi-factor operational condition rather than one
        isolated subsystem problem. Identify relevant
        multi-factor sustainability knowledge.
        """,
        "subsystem": None,
        "expected_topic": "Multi-factor",
    },

    {
        "name": "Data quality issue",
        "query": """
        Data-center telemetry contains missing or invalid
        measurements and potentially unreliable sustainability
        metrics. Identify relevant data-quality knowledge.
        """,
        "subsystem": None,
        "expected_topic": "Data quality",
    },
]


# -------------------------------------------------------------------
# Load retriever
# -------------------------------------------------------------------

retriever = SustainabilityRetriever()


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

top1_correct = 0
top3_correct = 0

results_summary = []


for i, case in enumerate(TEST_CASES, 1):

    print("\n" + "-" * 70)
    print(f"TEST {i}: {case['name']}")
    print("-" * 70)

    results = retriever.retrieve(
        case["query"],
        top_k=3,
        subsystem=case["subsystem"]
    )

    topics = [str(result["topic"]) for result in results]

    print("Expected topic:", case["expected_topic"])
    print("Retrieved topics:", topics)

    # ---------------------------------------------------------------
    # Top-1
    # ---------------------------------------------------------------

    top1 = (
        len(topics) > 0
        and topics[0].strip().lower()
        == case["expected_topic"].strip().lower()
    )

    # ---------------------------------------------------------------
    # Top-3
    # ---------------------------------------------------------------

    top3 = any(
        topic.strip().lower()
        == case["expected_topic"].strip().lower()
        for topic in topics
    )

    if top1:
        top1_correct += 1

    if top3:
        top3_correct += 1

    print("Top-1:", "PASS" if top1 else "FAIL")
    print("Top-3:", "PASS" if top3 else "FAIL")

    # Show both scores so reranking can be audited.
    for rank, result in enumerate(results, 1):

        print(
            f"  {rank}. "
            f"{result['topic']} | "
            f"semantic={result['score']:.4f} | "
            f"final={result['final_score']:.4f}"
        )

    results_summary.append({
        "name": case["name"],
        "expected": case["expected_topic"],
        "top1": top1,
        "top3": top3,
    })


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

total = len(TEST_CASES)

top1_accuracy = top1_correct / total
top3_recall = top3_correct / total


print("\n")
print("=" * 70)
print("RAG EVALUATION RESULTS")
print("=" * 70)

print(f"Test cases      : {total}")
print(f"Top-1 correct   : {top1_correct}/{total}")
print(f"Top-1 accuracy  : {top1_accuracy:.2%}")
print(f"Top-3 correct   : {top3_correct}/{total}")
print(f"Top-3 recall    : {top3_recall:.2%}")


print("\nDetailed summary:")

for result in results_summary:

    print(
        f"- {result['name']}: "
        f"Top-1={'PASS' if result['top1'] else 'FAIL'}, "
        f"Top-3={'PASS' if result['top3'] else 'FAIL'}"
    )


print("\n" + "=" * 70)
print("GREENGUARD RAG EVALUATION COMPLETE")
print("=" * 70)