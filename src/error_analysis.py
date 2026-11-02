import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/aiml_content_engagement.csv")
OUTPUT_PATH = Path("reports/error_analysis.csv")

df = pd.read_csv(DATA_PATH)

# Basic dataset analysis
print("\n========== DATASET ERROR ANALYSIS ==========")
print("Total interactions:", len(df))
print("Unique students:", df["student_id"].nunique())
print("Unique subjects:", df["subject"].nunique())
print("Unique content types:", df["content_type"].nunique())

# Engagement distribution
print("\n========== ENGAGEMENT DISTRIBUTION ==========")
print(df["engagement_score"].describe())

# Engagement by subject
print("\n========== SUBJECT-WISE PERFORMANCE ==========")
subject_analysis = (
    df.groupby("subject")["engagement_score"]
    .agg(["count", "mean", "min", "max"])
    .sort_values("mean", ascending=False)
)

print(subject_analysis)

# Engagement by content type
print("\n========== CONTENT-TYPE PERFORMANCE ==========")
content_analysis = (
    df.groupby("content_type")["engagement_score"]
    .agg(["count", "mean", "min", "max"])
    .sort_values("mean", ascending=False)
)

print(content_analysis)

# Student interaction sparsity
student_counts = df.groupby("student_id").size()

print("\n========== SPARSITY ANALYSIS ==========")
print("Students with only 1 interaction:", (student_counts == 1).sum())
print("Students with 2 or more interactions:", (student_counts >= 2).sum())
print("Students with 3 or more interactions:", (student_counts >= 3).sum())

# Save subject and content-type analysis
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

subject_analysis.to_csv("reports/error_analysis_by_subject.csv")
content_analysis.to_csv("reports/error_analysis_by_content_type.csv")

print("\nReports saved successfully:")
print("reports/error_analysis_by_subject.csv")
print("reports/error_analysis_by_content_type.csv")