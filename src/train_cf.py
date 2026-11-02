
import os
import joblib
import pandas as pd
import mlflow

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# 1. Load dataset
DATA_PATH = "data/raw/aiml_content_engagement.csv"

df = pd.read_csv(DATA_PATH)

# 2. Create proxy item IDs and ratings
df["item_id"] = (
    df["subject"].astype(str) + "|" +
    df["content_type"].astype(str)
)

# Convert engagement score (0.1–1.0) to a 1–5 rating scale
df["rating"] = 1 + (df["engagement_score"] * 4)

# 3. Prepare Surprise dataset
reader = Reader(rating_scale=(1, 5))

interaction_data = Dataset.load_from_df(
    df[["student_id", "item_id", "rating"]],
    reader
)

trainset, testset = train_test_split(
    interaction_data,
    test_size=0.2,
    random_state=42
)

# 4. Create MLflow experiment
mlflow.set_experiment("EduTrack_SVD_Experiments")

os.makedirs("reports", exist_ok=True)

# 5. Define two hyperparameter configurations
configurations = [
    {
        "n_factors": 20,
        "n_epochs": 20,
        "reg_all": 0.02
    },
    {
        "n_factors": 50,
        "n_epochs": 30,
        "reg_all": 0.05
    }
]

results = []

# 6. Train and evaluate both configurations
for index, params in enumerate(configurations, start=1):

    with mlflow.start_run(run_name=f"SVD_Config_{index}"):

        model = SVD(
            n_factors=params["n_factors"],
            n_epochs=params["n_epochs"],
            reg_all=params["reg_all"],
            random_state=42
        )

        # Train model
        model.fit(trainset)

        # Evaluate model
        predictions = model.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Save model using joblib and log as MLflow artifact
        model_path = f"reports/svd_model_config_{index}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Store results
        results.append({
            "configuration": index,
            "n_factors": params["n_factors"],
            "n_epochs": params["n_epochs"],
            "reg_all": params["reg_all"],
            "RMSE": rmse,
            "MAE": mae
        })

        print(f"\nSVD Configuration {index}")
        print(f"Parameters: {params}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("Model saved and logged successfully.")


# 7. Save experiment results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/svd_experiment_results.csv",
    index=False
)

# 8. Identify best configuration based on RMSE
best_result = results_df.loc[results_df["RMSE"].idxmin()]

print("\n========== EXPERIMENT RESULTS ==========")
print(results_df.to_string(index=False))

print("\n========== BEST CONFIGURATION ==========")
print(best_result.to_string())

print("\nExperiment completed successfully!")
