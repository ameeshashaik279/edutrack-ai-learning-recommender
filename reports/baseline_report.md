# EduTrack Baseline Report

## 1. Dataset Summary

- Student performance dataset: 1,000 rows and 16 columns.
- Content engagement dataset: 600 rows and 13 columns.
- Common column: `student_id`.
- Course-level identifiers are not available in the inspected datasets.

## 2. Data Pipeline

- Loaded datasets using Pandas.
- Standardized column names.
- Removed duplicate rows.
- Checked missing values.
- Student performance dataset: 0 duplicate rows and 0 missing values.
- Content engagement dataset: 0 duplicate rows and 0 missing values.

## 3. Baseline Model

- Built an engagement-based baseline using average scores grouped by subject and content type.
- Ranked subject and content-type combinations by average engagement score.
- Excluded combinations the student had already interacted with.
- Avoided recommending the same subject more than once.
- Generated up to five recommendations for a sample student.

## 4. Initial Test

- Test student: `STU-00698`.
- Generated suggestions included Digital Marketing assignments and Data Science practice problems.
- The recommender test ran successfully.

## 5. Evaluation Method

- Implemented Precision@K, Recall@K, and NDCG@K evaluation metrics.
- Used a chronological 80/20 train-test split, with earlier records for training and later records for testing.
- Used an engagement score threshold of 0.7 to identify relevant subjects.
- The relevance threshold is provisional and requires further validation.

## 6. Latest Chronological Evaluation

- Test rows: 120 out of 600 engagement records.
- Students evaluated: 36.
- Mean Precision@10: 0.100.
- Mean Recall@10: 1.000.
- Mean NDCG@10: 0.402.
- Target NDCG@10: 0.550.
- Status: Target not yet reached.

### Evaluation Limitations

- The relevance threshold of 0.7 is provisional.
- Student interactions are sparse, limiting reliable personalized evaluation.
- Recall@10 of 1.000 does not, by itself, demonstrate strong recommendation quality.
- The current evaluation measures subject relevance rather than course-level recommendation quality.
- Further model improvements and validation are required.

## 7. Interaction Data Analysis

- Total engagement records: 600.
- Unique students: 459.
- Average records per student: 1.31.
- Students with multiple records: 117.
- Students with three or more records: 23.
- Duplicate student-subject pairs: 12.

### Implications

- Most students have limited interaction history.
- Personalized recommendation evaluation is constrained by sparse data.
- More repeated student-course interactions are needed for reliable evaluation.
- The inspected datasets do not contain course-level identifiers, so the current baseline recommends subjects and content types rather than actual courses.

## 8. Next Steps

- Investigate and improve recommendation ranking.
- Validate the relevance threshold and evaluation methodology.
- Evaluate cold-start fallback behavior.
- Obtain course-level interaction data for genuine course recommendations.
- Re-evaluate performance against the NDCG@10 target of 0.550.
- Document model limitations and future improvements.

## 9. Conclusion

The data ingestion and preprocessing pipeline, baseline recommendation model, evaluation utilities, and baseline report have been implemented. The latest chronological evaluation produced a mean NDCG@10 of 0.402, which is below the target of 0.550. The current implementation is an initial baseline and requires further improvement and validation before the target can be considered achieved.