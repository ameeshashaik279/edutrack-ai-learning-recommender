"""
Evaluation utilities for the EduTrack recommendation system.
"""


def precision_at_k(recommended, relevant, k=10):
    """Calculate Precision@K."""
    recommended_k = recommended[:k]

    if not recommended_k:
        return 0.0

    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Calculate Recall@K."""
    recommended_k = recommended[:k]

    if not relevant:
        return 0.0

    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def evaluate_recommendations(recommended, relevant, k=10):
    """Evaluate recommendation results."""
    return {
        "precision_at_k": precision_at_k(recommended, relevant, k),
        "recall_at_k": recall_at_k(recommended, relevant, k),
    }


if __name__ == "__main__":
    print("EduTrack evaluation module initialized.")
