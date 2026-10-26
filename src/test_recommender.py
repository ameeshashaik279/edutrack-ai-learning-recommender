
import pandas as pd

from recommender import train_model, recommend_courses
from evaluation import evaluate_recommendations


# 1. Load the engagement dataset
data = pd.read_csv("data/raw/aiml_content_engagement.csv")


# 2. Standardize column names
data.columns = (
    data.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# 3. Validate required columns
required_columns = {
    "student_id",
    "subject",
    "content_type",
    "engagement_score",
    "date",
}

missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# 4. Clean the dataset
data["engagement_score"] = pd.to_numeric(
    data["engagement_score"],
    errors="coerce",
)

data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce",
)

data = data.dropna(
    subset=[
        "student_id",
        "subject",
        "content_type",
        "engagement_score",
        "date",
    ]
).copy()


if data.empty:
    raise ValueError("No valid engagement records available.")


# 5. Sort by date and split chronologically
data = data.sort_values("date").reset_index(drop=True)

split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Not enough records to create train/test sets.")

print("Training rows:", len(train_data))
print("Test rows:", len(test_data))

print(
    "Training date range:",
    train_data["date"].min(),
    "to",
    train_data["date"].max(),
)

print(
    "Testing date range:",
    test_data["date"].min(),
    "to",
    test_data["date"].max(),
)

print("Unique test students:", test_data["student_id"].nunique())


# 6. Train the baseline recommender using training data only
model = train_model(train_data)


# 7. Evaluate recommendations on later interactions
all_recommended = []
all_relevant = []

test_students = test_data["student_id"].dropna().unique()

for student_id in test_students:

    recommendations = recommend_courses(
        model,
        student_id,
        n=10,
    )

    recommended_subjects = list(dict.fromkeys(
        item["subject"] for item in recommendations
    ))

    student_test = test_data[
        test_data["student_id"] == student_id
    ]

    subject_scores = (
        student_test.groupby("subject")["engagement_score"].mean()
    )

    # Provisional relevance threshold
    relevant_subjects = subject_scores[
        subject_scores >= 0.7
    ].index.tolist()

    if not relevant_subjects:
        continue

    results = evaluate_recommendations(
        recommended_subjects,
        relevant_subjects,
        k=10,
    )

    all_recommended.append(recommended_subjects)
    all_relevant.append(relevant_subjects)

    print(
        f"{student_id}: "
        f"Precision@10={results['precision_at_k']:.3f}, "
        f"Recall@10={results['recall_at_k']:.3f}, "
        f"NDCG@10={results['ndcg_at_k']:.3f}"
    )


# 8. Calculate mean evaluation metrics
print("\nStudents evaluated:", len(all_recommended))

if all_recommended:

    metrics = [
        evaluate_recommendations(rec, rel, k=10)
        for rec, rel in zip(all_recommended, all_relevant)
    ]

    mean_precision = sum(
        item["precision_at_k"] for item in metrics
    ) / len(metrics)

    mean_recall = sum(
        item["recall_at_k"] for item in metrics
    ) / len(metrics)

    mean_ndcg = sum(
        item["ndcg_at_k"] for item in metrics
    ) / len(metrics)

    print("Mean Precision@10:", round(mean_precision, 3))
    print("Mean Recall@10:", round(mean_recall, 3))
    print("Mean NDCG@10:", round(mean_ndcg, 3))

    target_ndcg = 0.55

    if mean_ndcg >= target_ndcg:
        print("NDCG target status: REACHED")
    else:
        print("NDCG target status: NOT REACHED")
        print("Further model improvement is required.")

else:
    print("No test students met the relevance threshold of 0.7.")
    print("Evaluation is inconclusive.")
