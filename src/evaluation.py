
"""
Evaluation utilities for the EduTrack recommendation system.
"""


import math


def precision_at_k(recommended, relevant, k=10):
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Calculate Recall@K."""
    if not relevant:
        return 0.0

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(set(relevant))


def ndcg_at_k(recommended, relevant, k=10):
    """Calculate NDCG@K for binary relevance."""
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    recommended_k = recommended[:k]

    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(recommended_k)
        if item in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


def evaluate_recommendations(recommended, relevant, k=10):
    """Calculate Precision@K, Recall@K, and NDCG@K."""
    return {
        "precision_at_k": precision_at_k(recommended, relevant, k),
        "recall_at_k": recall_at_k(recommended, relevant, k),
        "ndcg_at_k": ndcg_at_k(recommended, relevant, k),
    }


if __name__ == "__main__":
    print("EduTrack evaluation module initialized.")

