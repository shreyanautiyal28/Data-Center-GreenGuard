import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

KNOWLEDGE_PATH = "rag/knowledge_base.csv"


class SustainabilityRetriever:

    def __init__(self):
        print("Loading sustainability knowledge base...")

        self.df = pd.read_csv(KNOWLEDGE_PATH)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        documents = (
            self.df["topic"].fillna("") + " | " +
            self.df["component"].fillna("") + " | " +
            self.df["condition"].fillna("") + " | " +
            self.df["explanation"].fillna("")
        ).tolist()

        print("Creating embeddings...")

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.index.add(
            embeddings.astype(np.float32)
        )

        print(f"Knowledge base loaded: {len(self.df)} records")

    def retrieve(self, query, top_k=3, subsystem=None):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Retrieve a larger candidate pool first
        candidate_k = min(7, len(self.df))

        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            candidate_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            row = self.df.iloc[index]

            semantic_score = float(score)

            # Start with semantic similarity
            final_score = semantic_score

            topic = str(row["topic"]).lower()
            component = str(row["component"]).lower()

            # Context-aware reranking
            if subsystem:

                subsystem_lower = str(subsystem).lower()

                if subsystem_lower == "hvac":
                    if "hvac" in topic or "hvac" in component:
                        final_score += 0.15

                elif subsystem_lower == "pump":
                    if "pump" in topic or "pump" in component:
                        final_score += 0.15

                elif subsystem_lower == "cooling":
                    if "cooling" in topic or "cooling" in component:
                        final_score += 0.15

                elif subsystem_lower == "it":
                    if (
                        topic == "low it load"
                        or component.strip() == "it"
                    ):
                        final_score += 0.15

            results.append({
                "score": semantic_score,
                "final_score": final_score,
                "topic": row["topic"],
                "component": row["component"],
                "condition": row["condition"],
                "explanation": row["explanation"],
                "recommended_action": row["recommended_action"],
                "sustainability_area": row["sustainability_area"]
            })

        # Sort using context-aware final score
        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results[:top_k]