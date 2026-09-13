from rag.retriever import SustainabilityRetriever


print("========================================")
print("GREENGUARD RAG RETRIEVAL TEST")
print("========================================")

retriever = SustainabilityRetriever()


query = """
Data-center anomaly with elevated HVAC-to-IT ratio,
high PUE and elevated pump-to-IT ratio.
Determine relevant sustainability knowledge
and operational investigation areas.
"""


print("\nQuery:")
print(query)

results = retriever.retrieve(
    query,
    top_k=3
)


print("\nRetrieved knowledge:")

for i, result in enumerate(results, 1):

    print("\n----------------------------------------")
    print(f"Result {i}")
    print("----------------------------------------")

    print("Similarity:", round(result["score"], 4))
    print("Topic:", result["topic"])
    print("Component:", result["component"])
    print("Condition:", result["condition"])
    print("Explanation:", result["explanation"])
    print("Recommended action:", result["recommended_action"])
    print("Sustainability area:", result["sustainability_area"])


print("\nRAG retrieval completed.")