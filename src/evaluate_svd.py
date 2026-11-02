
import os
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# 1. Load data
df = pd.read_csv("data/raw/aiml_content_engagement.csv")

df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

df["rating"] = 1 + (df["engagement_score"] * 4)

# 2. Prepare data
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    df[["student_id", "item_id", "rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# 3. Train best SVD model
model = SVD(
    n_factors=20,
    n_epochs=20,
    reg_all=0.02,
    random_state=42
)

model.fit(trainset)

# 4. Get items known to the training model
all_items = [
    trainset.to_raw_iid(i)
    for i in trainset.all_items()
]

# 5. Build training history
train_history = {}

for user_inner_id, item_inner_id, rating in trainset.all_ratings():

    user_id = trainset.to_raw_uid(user_inner_id)
    item_id = trainset.to_raw_iid(item_inner_id)

    train_history.setdefault(user_id, set()).add(item_id)

# 6. Build test relevance
test_df = pd.DataFrame(
    testset,
    columns=["student_id", "item_id", "rating"]
)

test_relevance = {}

for _, row in test_df.iterrows():

    user_id = row["student_id"]
    item_id = row["item_id"]
    rating = row["rating"]

    # Consider ratings >= 4 as relevant
    if rating >= 4:
        test_relevance.setdefault(user_id, {})[item_id] = max(
            rating,
            test_relevance.get(user_id, {}).get(item_id, 0)
        )

# 7. NDCG calculation
def calculate_ndcg(recommended_items, relevant_items, k=10):

    dcg = 0.0

    for position, item_id in enumerate(recommended_items[:k]):
        relevance = relevant_items.get(item_id, 0)

        dcg += (2 ** relevance - 1) / np.log2(position + 2)

    ideal_relevances = sorted(
        relevant_items.values(),
        reverse=True
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / np.log2(position + 2)
        for position, relevance in enumerate(ideal_relevances)
    )

    return dcg / idcg if idcg > 0 else 0.0


# 8. Evaluate unseen recommendations
ndcg_scores = []

for user_id, relevant_items in test_relevance.items():

    seen_items = train_history.get(user_id, set())

    candidates = [
        item_id for item_id in all_items
        if item_id not in seen_items
    ]

    if not candidates or not relevant_items:
        continue

    predictions = [
        (item_id, model.predict(user_id, item_id).est)
        for item_id in candidates
    ]

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_10 = [
        item_id for item_id, score in predictions[:10]
    ]

    score = calculate_ndcg(
        top_10,
        relevant_items,
        k=10
    )

    ndcg_scores.append(score)


# 9. Display results
mean_ndcg = (
    float(np.mean(ndcg_scores))
    if ndcg_scores else 0.0
)

print("\n========== UNSEEN ITEM EVALUATION ==========")
print("Students evaluated:", len(ndcg_scores))
print("NDCG@10:", round(mean_ndcg, 4))
print("Target:", 0.55)

if mean_ndcg >= 0.55:
    print("Target achieved in this evaluation.")
else:
    print("Target not achieved. Model improvement required.")


# 10. Save evaluation report
os.makedirs("reports", exist_ok=True)

result = pd.DataFrame([{
    "model": "SVD",
    "n_factors": 20,
    "n_epochs": 20,
    "reg_all": 0.02,
    "students_evaluated": len(ndcg_scores),
    "NDCG@10": mean_ndcg,
    "target": 0.55,
    "evaluation": "Unseen items known to training catalog"
}])

result.to_csv(
    "reports/svd_ndcg_evaluation.csv",
    index=False
)

print("\nReport saved successfully!")
