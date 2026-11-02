
import os
import joblib
import pandas as pd
import mlflow

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("data/raw/aiml_content_engagement.csv")

df["item_id"] = (
    df["subject"].astype(str) + "|" +
    df["content_type"].astype(str)
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

# 3. Define configurations
configurations = [
    {"n_factors": 10, "n_epochs": 20, "reg_all": 0.02},
    {"n_factors": 20, "n_epochs": 30, "reg_all": 0.05},
    {"n_factors": 30, "n_epochs": 40, "reg_all": 0.10}
]

# 4. Create MLflow experiment
mlflow.set_experiment("EduTrack_SVD_Tuning")

os.makedirs("reports", exist_ok=True)

results = []

# 5. Train and evaluate
for index, params in enumerate(configurations, start=1):

    with mlflow.start_run(run_name=f"Tuning_Config_{index}"):

        model = SVD(
            n_factors=params["n_factors"],
            n_epochs=params["n_epochs"],
            reg_all=params["reg_all"],
            random_state=42
        )

        model.fit(trainset)

        predictions = model.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        model_path = f"reports/tuned_svd_model_{index}.pkl"

        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        results.append({
            "configuration": index,
            **params,
            "RMSE": rmse,
            "MAE": mae
        })

        print(f"\nConfiguration {index}")
        print("Parameters:", params)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

# 6. Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/svd_tuning_results.csv",
    index=False
)

# 7. Display best configuration
best = results_df.loc[results_df["RMSE"].idxmin()]

print("\n========== TUNING RESULTS ==========")
print(results_df.to_string(index=False))

print("\n========== BEST CONFIGURATION ==========")
print(best.to_string())

print("\nTuning experiment completed successfully!")
