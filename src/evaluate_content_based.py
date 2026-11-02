
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/raw/aiml_content_engagement.csv"
OUTPUT_PATH = "reports/content_based_evaluation.csv"


def calculate_ndcg(recommendations, relevant_items, k=10):

    dcg = 0.0

    for rank, item in enumerate(recommendations[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(rank + 2)

    ideal_hits = min(len(relevant_items), k)

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


def prepare_features(train_df, all_items):

    features = train_df.groupby("item_id").agg(
        subject=("subject", "first"),
        content_type=("content_type", "first"),
        avg_engagement=("engagement_score", "mean"),
        avg_difficulty=("difficulty_rating", "mean"),
        completion_rate=("completed", "mean"),
        avg_time=("time_spent_mins", "mean")
    ).reset_index()

    catalog = pd.DataFrame({"item_id": all_items})

    features = catalog.merge(
        features,
        on="item_id",
        how="left"
    )

    features["subject"] = features["item_id"].str.split("|").str[0]
    features["content_type"] = features["item_id"].str.split("|").str[1]

    numeric_cols = [
        "avg_engagement",
        "avg_difficulty",
        "completion_rate",
        "avg_time"
    ]

    # Fill missing item features using training-set averages
    for col in numeric_cols:
        features[col] = features[col].fillna(
            features[col].mean()
        )

    categorical = pd.get_dummies(
        features[["subject", "content_type"]]
    )

    scaler = StandardScaler()

    numeric = scaler.fit_transform(
        features[numeric_cols]
    )

    matrix = np.hstack([
        categorical.to_numpy(dtype=float),
        numeric
    ])

    return features, matrix


def evaluate():

    df = pd.read_csv(DATA_PATH)

    df["student_id"] = df["student_id"].astype(str)

    df["item_id"] = (
        df["subject"].astype(str)
        + "|"
        + df["content_type"].astype(str)
    )

    df["rating"] = 1 + df["engagement_score"] * 4

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    all_items = sorted(df["item_id"].unique())

    features, matrix = prepare_features(
        train_df,
        all_items
    )

    item_index = {
        item: index
        for index, item in enumerate(features["item_id"])
    }

    results = []

    for student_id in test_df["student_id"].unique():

        history = train_df[
            train_df["student_id"] == student_id
        ]

        student_test = test_df[
            test_df["student_id"] == student_id
        ]

        if history.empty:
            continue

        relevant_items = set(
            student_test.loc[
                student_test["rating"] >= 4,
                "item_id"
            ]
        )

        if not relevant_items:
            continue

        vectors = []
        weights = []

        for _, row in history.iterrows():

            item = row["item_id"]

            if item in item_index:
                vectors.append(matrix[item_index[item]])
                weights.append(
                    max(float(row["engagement_score"]), 0.01)
                )

        if not vectors:
            continue

        profile = np.average(
            np.array(vectors),
            axis=0,
            weights=np.array(weights)
        )

        scores = cosine_similarity(
            profile.reshape(1, -1),
            matrix
        )[0]

        seen_items = set(history["item_id"])

        ranked_items = [
            features.iloc[index]["item_id"]
            for index in np.argsort(scores)[::-1]
            if features.iloc[index]["item_id"] not in seen_items
        ]

        ndcg = calculate_ndcg(
            ranked_items,
            relevant_items,
            k=10
        )

        results.append({
            "student_id": student_id,
            "relevant_items": len(relevant_items),
            "NDCG@10": ndcg
        })

    if not results:
        print("No eligible students found for evaluation.")
        return

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    mean_ndcg = results_df["NDCG@10"].mean()

    print("\nContent-Based Evaluation Results")
    print("--------------------------------")
    print("Students evaluated:", len(results_df))
    print(f"Mean NDCG@10: {mean_ndcg:.4f}")
    print("Target NDCG@10: 0.55")
    print("Target achieved:", mean_ndcg >= 0.55)
    print("Report saved:", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate()
