
import pandas as pd
from recommender import train_model, recommend_courses
from evaluation import evaluate_recommendations

data = pd.read_csv("data/raw/aiml_content_engagement.csv")

data.columns = (
    data.columns.str.strip().str.lower().str.replace(" ", "_")
)

# Split the data into training and test sets
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

model = train_model(train_data)

all_recommended = []
all_relevant = []

# Evaluate multiple students
test_students = test_data["student_id"].dropna().unique()[:20]

for student_id in test_students:
    recommendations = recommend_courses(model, student_id, n=10)

    recommended_subjects = list(dict.fromkeys(
        item["subject"] for item in recommendations
    ))

    student_test = test_data[test_data["student_id"] == student_id]

    subject_scores = (
        student_test.groupby("subject")["engagement_score"].mean()
    )

    relevant_subjects = subject_scores[
        subject_scores >= 0.7
    ].index.tolist()

    if relevant_subjects:
        results = evaluate_recommendations(
            recommended_subjects,
            relevant_subjects,
            k=10
        )

        all_recommended.append(recommended_subjects)
        all_relevant.append(relevant_subjects)

        print(
            f"{student_id}: "
            f"Precision@10={results['precision_at_k']:.3f}, "
            f"Recall@10={results['recall_at_k']:.3f}, "
            f"NDCG@10={results['ndcg_at_k']:.3f}"
        )

print("Students evaluated:", len(all_recommended))

if all_recommended:
    metrics = [
        evaluate_recommendations(rec, rel, k=10)
        for rec, rel in zip(all_recommended, all_relevant)
    ]

    print(
        "Mean Precision@10:",
        round(sum(m["precision_at_k"] for m in metrics) / len(metrics), 3)
    )
    print(
        "Mean Recall@10:",
        round(sum(m["recall_at_k"] for m in metrics) / len(metrics), 3)
    )
    print(
        "Mean NDCG@10:",
        round(sum(m["ndcg_at_k"] for m in metrics) / len(metrics), 3)
    )
else:
    print("No test students met the relevance threshold.")
