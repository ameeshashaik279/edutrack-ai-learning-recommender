
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/raw/aiml_content_engagement.csv")

df["date"] = pd.to_datetime(df["date"])
df["student_id"] = df["student_id"].astype(str)

df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

df["rating"] = 1 + df["engagement_score"] * 4


# ==========================================
# 2. CHRONOLOGICAL TRAIN TEST SPLIT
# ==========================================

counts = df.groupby("student_id").size()
eligible = counts[counts >= 2].index

df = df[df["student_id"].isin(eligible)]
df = df.sort_values("date")

train = df.groupby("student_id").head(-1).copy()
test = df.groupby("student_id").tail(1).copy()

catalog = sorted(train["item_id"].unique())

train_seen = train.groupby("student_id")["item_id"].apply(set)

relevant_test = test[test["engagement_score"] >= 0.75]

evaluation = []

for _, row in relevant_test.iterrows():

    user = row["student_id"]
    item = row["item_id"]

    if item in catalog and item not in train_seen[user]:
        evaluation.append((user, item))


print("\nDATASET SUMMARY")
print("----------------")
print("Eligible students:", len(eligible))
print("Training interactions:", len(train))
print("Testing interactions:", len(test))
print("Evaluation students:", len(evaluation))
print("Catalog items:", len(catalog))

if not evaluation:
    raise ValueError("No evaluation users found.")


# ==========================================
# 3. TRAIN SVD ON TRAINING DATA ONLY
# ==========================================

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    train[["student_id", "item_id", "rating"]],
    reader
)

trainset = data.build_full_trainset()

svd = SVD(
    n_factors=20,
    n_epochs=20,
    reg_all=0.02,
    random_state=42
)

svd.fit(trainset)


# ==========================================
# 4. POPULARITY MODEL
# ==========================================

popularity = (
    train.groupby("item_id")["engagement_score"]
    .mean()
    .to_dict()
)


# ==========================================
# 5. CONTENT-BASED FEATURES
# ==========================================

categorical_cols = ["subject", "content_type"]

numeric_cols = [
    "time_spent_mins",
    "difficulty_rating",
    "completed",
    "replayed",
    "paused_count",
    "liked",
    "notes_taken",
    "engagement_score"
]

item_features = train.groupby("item_id").agg({
    "subject": "first",
    "content_type": "first",
    "time_spent_mins": "mean",
    "difficulty_rating": "mean",
    "completed": "mean",
    "replayed": "mean",
    "paused_count": "mean",
    "liked": "mean",
    "notes_taken": "mean",
    "engagement_score": "mean"
})

item_features[numeric_cols] = (
    item_features[numeric_cols]
    .fillna(item_features[numeric_cols].mean())
)

categorical_features = pd.get_dummies(
    item_features[categorical_cols],
    dtype=float
)

scaler = StandardScaler()

numerical_features = scaler.fit_transform(
    item_features[numeric_cols]
)

feature_matrix = np.hstack([
    categorical_features.to_numpy(dtype=float),
    numerical_features
])

item_vectors = {
    item: feature_matrix[i]
    for i, item in enumerate(item_features.index)
}


# ==========================================
# 6. STUDENT PROFILES
# ==========================================

student_vectors = {}

for user, history in train.groupby("student_id"):

    vectors = []
    weights = []

    for _, row in history.iterrows():

        item = row["item_id"]

        if item in item_vectors:
            vectors.append(item_vectors[item])
            weights.append(
                max(float(row["engagement_score"]), 0.01)
            )

    if vectors:
        student_vectors[user] = np.average(
            np.vstack(vectors),
            axis=0,
            weights=weights
        )


# ==========================================
# 7. NORMALIZATION
# ==========================================

def normalize_scores(scores):

    values = np.array(list(scores.values()), dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return {key: 0.5 for key in scores}

    return {
        key: float((value - minimum) / (maximum - minimum))
        for key, value in scores.items()
    }


# ==========================================
# 8. NDCG@10
# ==========================================

def ndcg_at_10(ranked_items, relevant_item):

    top_items = ranked_items[:10]

    if relevant_item not in top_items:
        return 0.0

    rank = top_items.index(relevant_item) + 1

    return float(1 / np.log2(rank + 1))


# ==========================================
# 9. EVALUATE FOUR MODELS
# ==========================================

results = []

for user, relevant_item in evaluation:

    seen = train_seen[user]

    candidates = [
        item for item in catalog
        if item not in seen
    ]

    # Popularity scores
    pop_raw = {
        item: popularity.get(item, 0)
        for item in candidates
    }

    pop_scores = normalize_scores(pop_raw)

    # SVD scores
    svd_raw = {
        item: svd.predict(user, item).est
        for item in candidates
    }

    svd_scores = normalize_scores(svd_raw)

    # Content scores
    content_raw = {}

    student_vector = student_vectors[user].reshape(1, -1)

    for item in candidates:

        item_vector = item_vectors[item].reshape(1, -1)

        similarity = cosine_similarity(
            student_vector,
            item_vector
        )[0][0]

        content_raw[item] = float(similarity)

    content_scores = normalize_scores(content_raw)

    # Hybrid weighted combination
    hybrid_scores = {}

    for item in candidates:

        hybrid_scores[item] = (
            0.20 * pop_scores[item]
            + 0.40 * svd_scores[item]
            + 0.40 * content_scores[item]
        )

    # Ranking
    pop_ranked = sorted(
        candidates, key=lambda x: pop_scores[x], reverse=True
    )

    svd_ranked = sorted(
        candidates, key=lambda x: svd_scores[x], reverse=True
    )

    content_ranked = sorted(
        candidates, key=lambda x: content_scores[x], reverse=True
    )

    hybrid_ranked = sorted(
        candidates, key=lambda x: hybrid_scores[x], reverse=True
    )

    results.append({
        "student_id": user,
        "relevant_item": relevant_item,

        "Popularity_NDCG@10": ndcg_at_10(
            pop_ranked, relevant_item
        ),

        "SVD_NDCG@10": ndcg_at_10(
            svd_ranked, relevant_item
        ),

        "Content_NDCG@10": ndcg_at_10(
            content_ranked, relevant_item
        ),

        "Hybrid_NDCG@10": ndcg_at_10(
            hybrid_ranked, relevant_item
        )
    })


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/hybrid_evaluation.csv",
    index=False
)


# ==========================================
# 11. FINAL MODEL COMPARISON
# ==========================================

print("\n===================================")
print("       HYBRID MODEL EVALUATION")
print("===================================")

print("Evaluated students:", len(results_df))

columns = [
    "Popularity_NDCG@10",
    "SVD_NDCG@10",
    "Content_NDCG@10",
    "Hybrid_NDCG@10"
]

for column in columns:

    score = results_df[column].mean()

    print(column, ":", round(score, 4))

print("\nTarget NDCG@10: 0.55")

best_column = results_df[columns].mean().idxmax()

print("Best model:", best_column)

print("\nReport saved: reports/hybrid_evaluation.csv")

print("===================================")
