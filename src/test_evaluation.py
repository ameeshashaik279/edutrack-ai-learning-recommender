
from evaluation import evaluate_recommendations

recommended = ["Python", "SQL", "Power BI", "Java"]
relevant = ["Python", "SQL"]

results = evaluate_recommendations(
    recommended,
    relevant,
    k=4
)

print("Evaluation Results:")
print("Precision@4:", results["precision_at_k"])
print("Recall@4:", results["recall_at_k"])
