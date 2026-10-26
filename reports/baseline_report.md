
# EduTrack Baseline Report

## 1. Dataset Summary
- Student performance dataset: 1,000 rows and 16 columns.
- Content engagement dataset: 600 rows and 13 columns.
- Common column: student_id.
- Course-level identifiers are not available in the inspected datasets.

## 2. Data Pipeline
- Loaded datasets using Pandas.
- Standardized column names.
- Removed duplicate rows.
- Checked missing values.
- Student dataset check: 0 duplicate rows and 0 missing values.
- Engagement dataset check: 0 duplicate rows and 0 missing values.

## 3. Baseline Model
- Used average engagement scores grouped by subject and content type.
- Ranked groups by average engagement score.
- Generated up to five suggestions for a sample student.

## 4. Initial Test
- Test student: STU-00698
- Generated suggestions included Digital Marketing assignments and Data Science practice problems.
- The recommender test ran successfully.

## 5. Evaluation
- Sample Precision@4: 0.5
- Sample Recall@4: 1.0
- These are sample test results, not validated model performance.
- NDCG@10 has not yet been evaluated.

## 6. Limitations and Next Steps
- Current baseline recommends subject and content-type combinations, not actual courses.
- Course-level interaction data is needed for genuine course recommendations.
- Evaluate the model on held-out data.
- Implement and evaluate NDCG@10.
- Document the cold-start recommendation approach.
