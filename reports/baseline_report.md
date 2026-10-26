
# EduTrack Baseline Report

## 1. Dataset Summary

- Student performance dataset: 1,000 rows and 16 columns.
- Content engagement dataset: 600 rows and 13 columns.
- Common column: `student_id`.
- Course-level identifiers and ratings are not available in the inspected columns.

## 2. Data Pipeline

- Loaded both CSV datasets using Pandas.
- Standardized column names.
- Removed duplicate rows.
- Checked missing-value counts.
- The observed content engagement dataset run showed zero missing values.

## 3. Baseline Approach

- Grouped engagement scores by subject and content type.
- Ranked groups by average engagement score.
- Excluded subjects already present in the student's engagement history.

## 4. Initial Test

The baseline test generated up to five subject and content-type suggestions for a sample student.

Example suggestions included Digital Marketing assignments and Data Science practice problems.

## 5. Limitations

- This is a content-engagement baseline, not a validated course recommendation model.
- NDCG@10 has not yet been evaluated.
- Course-level interaction data is required for genuine personalized course recommendations.
- Precision and Recall test results should be recorded after running the evaluation test.
