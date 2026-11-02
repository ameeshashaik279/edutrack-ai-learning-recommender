
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "data/raw/aiml_content_engagement.csv"

df = pd.read_csv(DATA_PATH)

df["student_id"] = df["student_id"].astype(str)

df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

df["rating"] = 1 + df["engagement_score"] * 4


# ==========================================
# 2. TRAIN COLLABORATIVE FILTERING MODEL
# ==========================================

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    df[["student_id", "item_id", "rating"]],
    reader
)

trainset = data.build_full_trainset()

svd_model = SVD(
    n_factors=20,
    n_epochs=20,
    reg_all=0.02,
    random_state=42
)

svd_model.fit(trainset)


# ==========================================
# 3. BUILD ITEM FEATURES
# ==========================================

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

item_features = df.groupby("item_id").agg({
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

# Encode categorical columns
categorical_features = pd.get_dummies(
    item_features[categorical_cols],
    dtype=float
)

# Scale numerical columns
scaler = StandardScaler()

numerical_features = scaler.fit_transform(
    item_features[numeric_cols]
)

# Combine features
feature_matrix = np.hstack([
    categorical_features.to_numpy(dtype=float),
    numerical_features
])

item_vectors = {
    item: feature_matrix[i]
    for i, item in enumerate(item_features.index)
}


# ==========================================
# 4. BUILD STUDENT PROFILES
# ==========================================

student_vectors = {}

for user, history in df.groupby("student_id"):

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
# 5. POPULARITY SCORES
# ==========================================

popularity_scores = (
    df.groupby("item_id")["engagement_score"]
    .mean()
    .to_dict()
)


# ==========================================
# 6. SCORE NORMALIZATION
# ==========================================

def normalize_scores(scores):

    if not scores:
        return {}

    values = np.array(
        list(scores.values()),
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return {
            key: 0.5
            for key in scores
        }

    return {
        key: float((value - minimum) / (maximum - minimum))
        for key, value in scores.items()
    }


# ==========================================
# 7. HYBRID RECOMMENDATION FUNCTION
# ==========================================

def recommend_hybrid(student_id, top_n=5):

    student_id = str(student_id)

    # Check student history
    history = df[
        df["student_id"] == student_id
    ]

    # Items already viewed by student
    seen_items = set(history["item_id"])

    # Recommend only unseen catalog items
    candidates = [
        item for item in item_vectors
        if item not in seen_items
    ]

    if not candidates:
        print("No new recommendations available.")
        return pd.DataFrame()

    # --------------------------------------
    # COLD START: POPULARITY FALLBACK
    # --------------------------------------

    if student_id not in student_vectors:

        print("Cold-start user: Using popularity fallback.")

        ranked = sorted(
            candidates,
            key=lambda item: popularity_scores.get(item, 0),
            reverse=True
        )

        return format_recommendations(
            ranked[:top_n],
            popularity_scores
        )

    # --------------------------------------
    # A. POPULARITY SCORES
    # --------------------------------------

    pop_raw = {
        item: popularity_scores.get(item, 0)
        for item in candidates
    }

    pop_normalized = normalize_scores(pop_raw)

    # --------------------------------------
    # B. SVD SCORES
    # --------------------------------------

    svd_raw = {
        item: svd_model.predict(
            student_id,
            item
        ).est
        for item in candidates
    }

    svd_normalized = normalize_scores(svd_raw)

    # --------------------------------------
    # C. CONTENT-BASED SCORES
    # --------------------------------------

    content_raw = {}

    student_vector = student_vectors[
        student_id
    ].reshape(1, -1)

    for item in candidates:

        item_vector = item_vectors[
            item
        ].reshape(1, -1)

        similarity = cosine_similarity(
            student_vector,
            item_vector
        )[0][0]

        content_raw[item] = float(similarity)

    content_normalized = normalize_scores(content_raw)

    # --------------------------------------
    # D. COMBINE MODEL SCORES
    # --------------------------------------

        # --------------------------------------
    # D. COMBINE MODEL SCORES
    # --------------------------------------

    hybrid_scores = {}

    for item in candidates:
        hybrid_scores[item] = (
            0.70 * pop_normalized[item]
            + 0.20 * svd_normalized[item]
            + 0.10 * content_normalized[item]
        )

    ranked_items = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )

    return format_recommendations(
        ranked_items[:top_n],
        hybrid_scores
    )


# ==========================================
# 8. FORMAT RECOMMENDATIONS
# ==========================================

def format_recommendations(items, scores):

    output = []

    for item in items:

        subject, content_type = item.split("|", 1)

        output.append({
            "subject": subject,
            "content_type": content_type,
            "hybrid_score": round(
                float(scores.get(item, 0)),
                4
            )
        })

    return pd.DataFrame(output)


# ==========================================
# 9. RUN MODEL
# ==========================================

if __name__ == "__main__":

    student_id = "STU-00154"

    print("\nHYBRID RECOMMENDATION SYSTEM")
    print("================================")

    print("Student ID:", student_id)

    recommendations = recommend_hybrid(
        student_id,
        top_n=5
    )

    print("\nTop 5 Recommendations:\n")

    print(
        recommendations.to_string(index=False)
    )

    print("\n================================")
