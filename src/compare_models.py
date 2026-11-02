
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


# ==================================================
# 1. LOAD DATASET
# ==================================================

DATA_PATH = "data/raw/aiml_content_engagement.csv"
REPORT_PATH = "reports/fair_model_comparison.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df["student_id"] = df["student_id"].astype(str)

# Create item ID
df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

# Convert engagement score to rating
df["rating"] = 1 + df["engagement_score"] * 4


# ==================================================
# 2. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==================================================

counts = df.groupby("student_id").size()

eligible_users = counts[counts >= 2].index

df = df[df["student_id"].isin(eligible_users)]
df = df.sort_values("date")

# Older interactions for training
train = df.groupby("student_id").head(-1).copy()

# Latest interaction for testing
test = df.groupby("student_id").tail(1).copy()

catalog = sorted(train["item_id"].unique())

train_seen = (
    train.groupby("student_id")["item_id"]
    .apply(set)
)


# ==================================================
# 3. SELECT RELEVANT TEST INTERACTIONS
# ==================================================

relevant_test = test[
    test["engagement_score"] >= 0.75
].copy()

evaluation = []

for _, row in relevant_test.iterrows():

    user = row["student_id"]
    item = row["item_id"]

    # Item must exist in training catalog
    # and must not already be seen by the student

    if item in catalog and item not in train_seen[user]:
        evaluation.append((user, item))


print("\nDATASET SPLIT SUMMARY")
print("---------------------")
print("Eligible students:", len(eligible_users))
print("Training interactions:", len(train))
print("Testing interactions:", len(test))
print("High-engagement test interactions:", len(relevant_test))
print("Actually evaluable students:", len(evaluation))
print("Training catalog items:", len(catalog))


if not evaluation:
    raise ValueError("No evaluable students found.")


# ==================================================
# 4. POPULARITY BASELINE
# ==================================================

popularity = (
    train.groupby("item_id")["engagement_score"]
    .mean()
    .to_dict()
)


# ==================================================
# 5. TRAIN SVD COLLABORATIVE FILTERING MODEL
# ==================================================

reader = Reader(rating_scale=(1, 5))

surprise_data = Dataset.load_from_df(
    train[["student_id", "item_id", "rating"]],
    reader
)

trainset = surprise_data.build_full_trainset()

svd = SVD(
    n_factors=20,
    n_epochs=20,
    reg_all=0.02,
    random_state=42
)

svd.fit(trainset)

print("\nSVD model training completed.")


# ==================================================
# 6. CONTENT-BASED FEATURE ENGINEERING
# ==================================================

categorical_cols = [
    "subject",
    "content_type"
]

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

# Build item profiles using TRAINING DATA ONLY

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

# Fill missing numerical values
item_features[numeric_cols] = (
    item_features[numeric_cols]
    .fillna(item_features[numeric_cols].mean())
)

# One-hot encode categorical features
categorical_features = pd.get_dummies(
    item_features[categorical_cols],
    dtype=float
)

# Scale numerical features
scaler = StandardScaler()

numerical_features = scaler.fit_transform(
    item_features[numeric_cols]
)

# Combine both feature groups
feature_matrix = np.hstack([
    categorical_features.to_numpy(dtype=float),
    numerical_features
])

# Store one feature vector per item
item_vectors = {}

for index, item in enumerate(item_features.index):
    item_vectors[item] = feature_matrix[index]


# ==================================================
# 7. BUILD STUDENT PROFILES
# ==================================================

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


print("Content-based feature engineering completed.")


# ==================================================
# 8. NDCG@10 FUNCTION
# ==================================================

def ndcg_at_10(ranked_items, relevant_item):

    top_items = ranked_items[:10]

    if relevant_item not in top_items:
        return 0.0

    rank = top_items.index(relevant_item) + 1

    return float(1.0 / np.log2(rank + 1))


# ==================================================
# 9. EVALUATE THREE MODELS
# ==================================================

results = []

for user, relevant_item in evaluation:

    seen_items = train_seen[user]

    # Only unseen items from training catalog
    candidates = [
        item for item in catalog
        if item not in seen_items
    ]

    if relevant_item not in candidates:
        continue

    # ----------------------------------------------
    # A. POPULARITY MODEL
    # ----------------------------------------------

    popularity_ranked = sorted(
        candidates,
        key=lambda item: popularity.get(item, 0),
        reverse=True
    )

    # ----------------------------------------------
    # B. SVD MODEL
    # ----------------------------------------------

    svd_scores = {}

    for item in candidates:

        prediction = svd.predict(user, item)

        svd_scores[item] = prediction.est

    svd_ranked = sorted(
        candidates,
        key=lambda item: svd_scores[item],
        reverse=True
    )

    # ----------------------------------------------
    # C. CONTENT-BASED MODEL
    # ----------------------------------------------

    if user in student_vectors:

        content_scores = {}

        for item in candidates:

            student_vector = student_vectors[user].reshape(1, -1)

            item_vector = item_vectors[item].reshape(1, -1)

            similarity = cosine_similarity(
                student_vector,
                item_vector
            )[0][0]

            content_scores[item] = float(similarity)

        content_ranked = sorted(
            candidates,
            key=lambda item: content_scores[item],
            reverse=True
        )

    else:

        content_ranked = popularity_ranked

    # ----------------------------------------------
    # CALCULATE NDCG
    # ----------------------------------------------

    results.append({

        "student_id": user,

        "relevant_item": relevant_item,

        "Popularity_NDCG@10": ndcg_at_10(
            popularity_ranked,
            relevant_item
        ),

        "SVD_NDCG@10": ndcg_at_10(
            svd_ranked,
            relevant_item
        ),

        "Content_NDCG@10": ndcg_at_10(
            content_ranked,
            relevant_item
        )
    })


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    REPORT_PATH,
    index=False
)


# ==================================================
# 11. DISPLAY FINAL COMPARISON
# ==================================================

print("\n===================================")
print("       FAIR MODEL COMPARISON")
print("===================================")

print("Evaluated students:", len(results_df))

pop_score = results_df["Popularity_NDCG@10"].mean()

svd_score = results_df["SVD_NDCG@10"].mean()

content_score = results_df["Content_NDCG@10"].mean()

print("\nPopularity NDCG@10:", round(pop_score, 4))

print("SVD NDCG@10:", round(svd_score, 4))

print("Content-Based NDCG@10:", round(content_score, 4))

print("\nTarget NDCG@10: 0.55")

print("\nTARGET STATUS")

best_score = max(
    pop_score,
    svd_score,
    content_score
)

if best_score >= 0.55:
    print("Target achieved.")
else:
    print("Target not achieved yet.")

print("\nBest model:")

scores = {
    "Popularity": pop_score,
    "SVD": svd_score,
    "Content-Based": content_score
}

print(max(scores, key=scores.get))

print("\nReport saved:", REPORT_PATH)

print("===================================")
