
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/raw/aiml_content_engagement.csv"


def load_data():
    """Load and prepare engagement dataset."""

    df = pd.read_csv(DATA_PATH)

    df["student_id"] = df["student_id"].astype(str)

    df["item_id"] = (
        df["subject"].astype(str)
        + "|"
        + df["content_type"].astype(str)
    )

    return df


def build_item_features(df):
    """Create numerical features for each subject-content item."""

    item_features = df.groupby("item_id").agg(
        subject=("subject", "first"),
        content_type=("content_type", "first"),
        avg_engagement=("engagement_score", "mean"),
        avg_difficulty=("difficulty_rating", "mean"),
        completion_rate=("completed", "mean"),
        avg_time=("time_spent_mins", "mean")
    ).reset_index()

    # Encode categorical features
    categorical = pd.get_dummies(
        item_features[["subject", "content_type"]]
    )

    # Scale numerical features
    numeric_columns = [
        "avg_engagement",
        "avg_difficulty",
        "completion_rate",
        "avg_time"
    ]

    numeric = item_features[numeric_columns].fillna(0)

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)

    # Combine all features
    feature_matrix = np.hstack([
        categorical.to_numpy(dtype=float),
        numeric_scaled
    ])

    return item_features, feature_matrix


def recommend_content(student_id, top_n=5):

    df = load_data()

    student_id = str(student_id)

    item_features, feature_matrix = build_item_features(df)

    item_ids = item_features["item_id"].tolist()

    item_index = {
        item: index
        for index, item in enumerate(item_ids)
    }

    # Get student interaction history
    history = df[df["student_id"] == student_id]

    # Cold-start fallback
    if history.empty:

        recommendations = item_features.sort_values(
            by="avg_engagement",
            ascending=False
        ).head(top_n)

        print(
            f"\nCold-start recommendations for student {student_id}:"
        )

        print(
            recommendations[
                ["subject", "content_type", "avg_engagement"]
            ].to_string(index=False)
        )

        return recommendations

    # Build student profile
    profile_vectors = []
    weights = []

    for _, row in history.iterrows():

        item_id = row["item_id"]

        if item_id in item_index:

            index = item_index[item_id]

            profile_vectors.append(
                feature_matrix[index]
            )

            weights.append(
                max(float(row["engagement_score"]), 0.01)
            )

    # Fallback if no valid profile exists
    if not profile_vectors:

        recommendations = item_features.sort_values(
            by="avg_engagement",
            ascending=False
        ).head(top_n)

        print("\nPopularity-based fallback recommendations:")

        print(
            recommendations[
                ["subject", "content_type", "avg_engagement"]
            ].to_string(index=False)
        )

        return recommendations

    # Weighted average student profile
    student_profile = np.average(
        np.array(profile_vectors),
        axis=0,
        weights=np.array(weights)
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        student_profile.reshape(1, -1),
        feature_matrix
    )[0]

    results = item_features.copy()

    results["similarity_score"] = similarity_scores

    # Remove previously seen items
    seen_items = set(history["item_id"])

    results = results[
        ~results["item_id"].isin(seen_items)
    ]

    # Sort recommendations
    recommendations = results.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    print(
        f"\nPersonalized recommendations for student {student_id}:"
    )

    print(
        recommendations[
            ["subject", "content_type", "similarity_score"]
        ].to_string(index=False)
    )

    return recommendations


if __name__ == "__main__":

    # Student with existing interaction history
    student_id = "STU-00154"

    recommendations = recommend_content(
        student_id=student_id,
        top_n=5
    )
