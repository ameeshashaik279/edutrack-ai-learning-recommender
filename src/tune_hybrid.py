
import os
import warnings
import numpy as np
import pandas as pd

from math import log2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from surprise import Dataset, Reader, SVD

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/raw/aiml_content_engagement.csv"

df = pd.read_csv(DATA_PATH)

required_columns = [
    "student_id",
    "subject",
    "content_type",
    "engagement_score"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# Find date column
date_col = None

for col in df.columns:
    if any(word in col.lower() for word in ["date", "time", "timestamp"]):
        parsed_dates = pd.to_datetime(df[col], errors="coerce")

        if parsed_dates.notna().sum() > 0:
            df[col] = parsed_dates
            date_col = col
            break

if date_col is None:
    raise ValueError(
        "Date column not found. Please check dataset column names."
    )

df = df.dropna(
    subset=["student_id", "engagement_score", date_col]
).copy()

df["item_id"] = (
    df["subject"].astype(str)
    + "|"
    + df["content_type"].astype(str)
)

df = df.sort_values(date_col).reset_index(drop=True)

print("\nDATASET SUMMARY")
print("Total interactions:", len(df))
print("Unique students:", df["student_id"].nunique())
print("Unique items:", df["item_id"].nunique())


# --------------------------------------------------
# 2. OUTER TRAIN DATA
# Keep the latest interaction for final testing.
# Do not use final test data for weight tuning.
# --------------------------------------------------

eligible = df.groupby("student_id").filter(
    lambda x: len(x) >= 2
).copy()

eligible = eligible.sort_values(date_col).copy()

group_sizes = eligible.groupby("student_id")["student_id"].transform("size")
group_order = eligible.groupby("student_id").cumcount()

outer_train = eligible[
    group_order < group_sizes - 1
].copy()

print("\nOUTER TRAINING DATA")
print("Training interactions:", len(outer_train))
print("Eligible students:", outer_train["student_id"].nunique())


# --------------------------------------------------
# 3. INNER CHRONOLOGICAL VALIDATION SPLIT
# --------------------------------------------------

outer_train = outer_train.sort_values(date_col).copy()

inner_sizes = outer_train.groupby("student_id")["student_id"].transform("size")

inner_eligible = outer_train[
    outer_train["student_id"].map(
        outer_train["student_id"].value_counts()
    ) >= 2
].copy()

inner_eligible = inner_eligible.sort_values(date_col).copy()

inner_sizes = inner_eligible.groupby("student_id")["student_id"].transform("size")
inner_order = inner_eligible.groupby("student_id").cumcount()

inner_train = inner_eligible[
    inner_order < inner_sizes - 1
].copy()

validation = inner_eligible[
    inner_order == inner_sizes - 1
].copy()

inner_train = inner_train.reset_index(drop=True)
validation = validation.reset_index(drop=True)

print("\nVALIDATION SPLIT")
print("Inner training interactions:", len(inner_train))
print("Validation interactions:", len(validation))
print("Validation students:", validation["student_id"].nunique())


# --------------------------------------------------
# 4. FILTER HIGH-ENGAGEMENT VALIDATION DATA
# --------------------------------------------------

validation = validation[
    validation["engagement_score"] >= 0.75
].copy()

print(
    "High-engagement validation interactions:",
    len(validation)
)


# --------------------------------------------------
# 5. BUILD MODELS USING INNER TRAIN ONLY
# --------------------------------------------------

if inner_train.empty:
    raise ValueError(
        "Not enough training data to build models."
    )

catalog = sorted(inner_train["item_id"].unique())

seen_items = (
    inner_train.groupby("student_id")["item_id"]
    .apply(set)
    .to_dict()
)

# Popularity model
popularity = (
    inner_train.groupby("item_id")["engagement_score"]
    .mean()
    .to_dict()
)

# SVD model
svd_data = inner_train[
    ["student_id", "item_id", "engagement_score"]
].copy()

svd_data["rating"] = (
    1 + svd_data["engagement_score"] * 4
)

reader = Reader(rating_scale=(1, 5))

surprise_data = Dataset.load_from_df(
    svd_data[["student_id", "item_id", "rating"]],
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


# --------------------------------------------------
# 6. CONTENT-BASED FEATURE ENGINEERING
# --------------------------------------------------

categorical_cols = [
    "subject",
    "content_type"
]

numeric_cols = [
    col
    for col in inner_train.select_dtypes(include=np.number).columns
    if col not in ["engagement_score"]
]

# Aggregate numeric features by item
if numeric_cols:
    item_numeric = (
        inner_train.groupby("item_id")[numeric_cols]
        .mean()
        .reset_index()
    )
else:
    item_numeric = pd.DataFrame(
        {"item_id": inner_train["item_id"].unique()}
    )

# Get categorical item details
item_categories = (
    inner_train.groupby("item_id")[categorical_cols]
    .first()
    .reset_index()
)

item_features = item_numeric.merge(
    item_categories,
    on="item_id",
    how="left"
)

# Fill missing numerical values
for col in numeric_cols:
    item_features[col] = item_features[col].fillna(
        item_features[col].mean()
    )

transformers = []

if numeric_cols:
    transformers.append(
        (
            "numeric",
            StandardScaler(),
            numeric_cols
        )
    )

transformers.append(
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    )
)

preprocessor = ColumnTransformer(transformers)

feature_matrix = preprocessor.fit_transform(item_features)

if hasattr(feature_matrix, "toarray"):
    feature_matrix = feature_matrix.toarray()

item_vectors = {
    item: feature_matrix[index]
    for index, item in enumerate(item_features["item_id"])
}

print("Content-based feature engineering completed.")


# --------------------------------------------------
# 7. NORMALIZATION
# --------------------------------------------------

def normalize(scores):

    if not scores:
        return {}

    values = np.array(
        list(scores.values()),
        dtype=float
    )

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return {
            key: 0.5
            for key in scores
        }

    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in scores.items()
    }


# --------------------------------------------------
# 8. GET MODEL SCORES FOR A STUDENT
# --------------------------------------------------

def get_model_scores(student_id):

    seen = seen_items.get(student_id, set())

    candidates = [
        item
        for item in catalog
        if item not in seen
    ]

    if not candidates:
        return {}, {}, {}

    # Popularity scores
    pop_scores = {
        item: popularity.get(item, 0)
        for item in candidates
    }

    # SVD scores
    svd_scores = {
        item: svd.predict(student_id, item).est
        for item in candidates
    }

    # Content-based scores
    history = inner_train[
        inner_train["student_id"] == student_id
    ]

    content_scores = {}

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

        student_profile = np.average(
            vectors,
            axis=0,
            weights=weights
        ).reshape(1, -1)

        for item in candidates:

            if item in item_vectors:

                item_vector = item_vectors[item].reshape(1, -1)

                similarity = cosine_similarity(
                    student_profile,
                    item_vector
                )[0][0]

                content_scores[item] = float(similarity)

            else:
                content_scores[item] = 0.0

    else:

        content_scores = {
            item: 0.0
            for item in candidates
        }

    return (
        normalize(pop_scores),
        normalize(svd_scores),
        normalize(content_scores)
    )


# --------------------------------------------------
# 9. NDCG@10
# --------------------------------------------------

def ndcg_at_10(ranked_items, relevant_item):

    top_items = ranked_items[:10]

    if relevant_item not in top_items:
        return 0.0

    rank = top_items.index(relevant_item) + 1

    return 1 / log2(rank + 1)


# --------------------------------------------------
# 10. HYBRID WEIGHT TUNING
# Validation data only
# --------------------------------------------------

weight_options = [

    (0.70, 0.20, 0.10),

    (0.60, 0.30, 0.10),

    (0.60, 0.20, 0.20),

    (0.50, 0.30, 0.20),

    (0.50, 0.20, 0.30),

    (0.40, 0.30, 0.30),

    (0.30, 0.40, 0.30),

    (0.80, 0.10, 0.10),

    (0.70, 0.10, 0.20)

]

results = []

for weights in weight_options:

    wp, ws, wc = weights

    ndcg_scores = []

    for _, row in validation.iterrows():

        student_id = row["student_id"]
        relevant_item = row["item_id"]

        pop, svd_score, content = get_model_scores(
            student_id
        )

        # Relevant item must be in candidate catalog
        if relevant_item not in pop:
            continue

        combined_scores = {

            item: (
                wp * pop.get(item, 0)
                + ws * svd_score.get(item, 0)
                + wc * content.get(item, 0)
            )

            for item in pop
        }

        ranked_items = sorted(
            combined_scores,
            key=combined_scores.get,
            reverse=True
        )

        score = ndcg_at_10(
            ranked_items,
            relevant_item
        )

        ndcg_scores.append(score)

    mean_ndcg = (
        float(np.mean(ndcg_scores))
        if ndcg_scores
        else 0.0
    )

    results.append({

        "popularity_weight": wp,

        "svd_weight": ws,

        "content_weight": wc,

        "validation_NDCG@10": mean_ndcg,

        "evaluated_users": len(ndcg_scores)

    })


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

os.makedirs(
    "reports",
    exist_ok=True
)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="validation_NDCG@10",
    ascending=False
)

report_path = "reports/hybrid_weight_tuning.csv"

results_df.to_csv(
    report_path,
    index=False
)


# --------------------------------------------------
# 12. DISPLAY BEST WEIGHTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("HYBRID WEIGHT TUNING RESULTS")
print("=" * 55)

print(
    results_df.to_string(index=False)
)

if results_df.empty:

    print("\nNo weight results available.")

elif results_df["evaluated_users"].max() == 0:

    print(
        "\nNot enough validation data to select reliable weights."
    )

else:

    best = results_df.iloc[0]

    print("\nBEST VALIDATION WEIGHTS")

    print(
        "Popularity weight:",
        best["popularity_weight"]
    )

    print(
        "SVD weight:",
        best["svd_weight"]
    )

    print(
        "Content weight:",
        best["content_weight"]
    )

    print(
        "Validation NDCG@10:",
        round(best["validation_NDCG@10"], 4)
    )

    print(
        "Evaluated users:",
        int(best["evaluated_users"])
    )

print("\nReport saved:", report_path)

print(
    "\nIMPORTANT: These are validation results, "
    "not final test results."
)
