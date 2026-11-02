
import os
import numpy as np
import pandas as pd
import mlflow

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# 1. Load dataset
df = pd.read_csv("data/raw/aiml_content_engagement.csv")

df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

df["rating"] = 1 + (df["engagement_score"] * 4)

# 2. Prepare dataset
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

# 3. Build training history and test relevance
all_items = [
    trainset.to_raw_iid(i)
    for i in trainset.all_items()
]

train_history = {}

for uid, iid, rating in trainset.all_ratings():
    user_id = trainset.to_raw_uid(uid)
    item_id = trainset.to_raw_iid(iid)
    train_history.setdefault(user_id, set()).add(item_id)

test_df = pd.DataFrame(
    testset,
    columns=["student_id", "item_id", "rating"]
)

test_relevance = {}

for _, row in test_df.iterrows():
    user_id = row["student_id"]
    item_id = row["item_id"]
    rating = row["rating"]

    if rating >= 4:
        test_relevance.setdefault(user_id, {})[item_id] = max(
            rating,
            test_relevance.get(user_id, {}).get(item_id, 0)
        )


# 4. NDCG@10 function
def calculate_ndcg(recommended, relevant, k=10):

    dcg = sum(
        (2 ** relevant.get(item, 0) - 1)
        / np.log2(position + 2)
        for position, item in enumerate(recommended[:k])
    )

    ideal = sorted(relevant.values(), reverse=True)[:k]

    idcg = sum(
        (2 ** rating - 1) / np.log2(position + 2)
        for position, rating in enumerate(ideal)
    )

    return dcg / idcg if idcg > 0 else 0.0


# 5. Three configurations
configurations = [
    {"n_factors": 10, "n_epochs": 20, "reg_all": 0.02},
    {"n_factors": 20, "n_epochs": 30, "reg_all": 0.05},
    {"n_factors": 30, "n_epochs": 40, "reg_all": 0.10}
]

mlflow.set_experiment("EduTrack_SVD_Ranking_Comparison")

results = []

# 6. Train and evaluate each configuration
for index, params in enumerate(configurations, start=1):

    model = SVD(
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        reg_all=params["reg_all"],
        random_state=42
    )

    model.fit(trainset)

    scores = []

    for user_id, relevant in test_relevance.items():

        seen = train_history.get(user_id, set())

        candidates = [
            item for item in all_items
            if item not in seen
        ]

        if not candidates:
            continue

        predictions = [
            (item, model.predict(user_id, item).est)
            for item in candidates
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        top_10 = [item for item, _ in predictions[:10]]

        scores.append(
            calculate_ndcg(top_10, relevant, k=10)
        )

    mean_ndcg = float(np.mean(scores)) if scores else 0.0

    with mlflow.start_run(run_name=f"Ranking_Config_{index}"):

        mlflow.log_params(params)
        mlflow.log_metric("NDCG_at_10", mean_ndcg)
        mlflow.log_metric("students_evaluated", len(scores))

    results.append({
        "configuration": index,
        **params,
        "students_evaluated": len(scores),
        "NDCG@10": mean_ndcg
    })

    print(f"\nConfiguration {index}")
    print("Parameters:", params)
    print("Students evaluated:", len(scores))
    print(f"NDCG@10: {mean_ndcg:.4f}")


# 7. Save comparison report
os.makedirs("reports", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/svd_ndcg_comparison.csv",
    index=False
)

print("\n========== NDCG COMPARISON ==========")
print(results_df.to_string(index=False))

best = results_df.loc[results_df["NDCG@10"].idxmax()]

print("\n========== BEST RANKING CONFIGURATION ==========")
print(best.to_string())

print("\nTarget NDCG@10: 0.55")
print("\nReport saved: reports/svd_ndcg_comparison.csv")
