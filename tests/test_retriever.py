from rag.retriever import SustainabilityRetriever


def main():

    print("\n🌱 GreenGuard RAG Test")
    print("=" * 50)

    retriever = SustainabilityRetriever()

    query = (
        "I drive my car for short trips every day "
        "and want to reduce my environmental impact."
    )

    results = retriever.search(
        query,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieved Knowledge:")
    print("-" * 50)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Issue: "
            f"{result['issue']}"
        )

        print(
            f"Recommendation: "
            f"{result['recommendation']}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

        print(
            f"Impact: "
            f"{result['impact_level']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity_score']:.4f}"
        )


if __name__ == "__main__":
    main()