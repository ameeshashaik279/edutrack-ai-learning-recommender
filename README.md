# EduTrack AI Learning Recommender

An adaptive learning path recommendation system for an EdTech platform.

## Project Objective

The goal of this project is to recommend the next 5 courses for a student based on previous learning interactions and engagement patterns.

## Proposed Approach

The system will use collaborative filtering with matrix factorization to generate personalized course recommendations.

A content-based fallback will be used for cold-start students who have insufficient interaction history.

## Technology Stack

- Python
- Pandas
- NumPy
- Surprise
- Scikit-learn
- FastAPI
- Streamlit
- Git/GitHub

## System Architecture

Student Data
    ↓
Data Cleaning and Preprocessing
    ↓
Student-Course Interaction Matrix
    ↓
Collaborative Filtering
    ↓
Matrix Factorization / SVD
    ↓
Top-5 Course Recommendations
    ↓
FastAPI
    ↓
Streamlit Dashboard

Cold-start students:
New Student
    ↓
Content-Based Fallback
    ↓
Course Recommendations

## Evaluation Metrics

Primary metric:

- NDCG@10

Target:

- NDCG@10 >= 0.55

Additional metrics:

- Precision@K
- Recall@K
- RMSE / MAE where applicable
- Recommendation latency

Dashboard target:

- Recommendations should render in less than 1 second.

## Project Milestones

### Week 1
- Research relevant recommendation-system approaches
- Define system architecture
- Define evaluation metrics
- Set up repository and data pipeline skeleton

### Week 3
- Data preprocessing
- Data pipeline
- Baseline recommendation model

### Week 6
- Collaborative filtering model
- Experimentation
- Model improvement

### Week 9
- FastAPI integration
- Streamlit dashboard
- Testing

### Week 12
- Final demo
- Model card
- Final documentation

## Dataset

The project uses the provided student performance and content engagement datasets.

The datasets will be explored and transformed into the interaction representation required for recommendation modeling.

## Status

## Week 1 Research

### 1. Matrix Factorization

Matrix factorization is a collaborative filtering technique used to learn relationships between students and courses. It will be considered as the main recommendation approach for this project.

Reference:
https://doi.org/10.1109/MC.2009.263

### 2. Surprise Library

Surprise is a Python library for building recommendation systems. It provides algorithms such as SVD for matrix factorization.

Reference:
https://github.com/NicolasHug/Surprise

### 3. Cold-Start Recommendation

Cold-start handling is important when a new student has little or no interaction history. A content-based fallback will be considered for such students.

Reference:
https://ceur-ws.org/Vol-1448/paper4.pdf

## Research Findings

The proposed system will use collaborative filtering with matrix factorization as the primary recommendation approach. Surprise SVD will be evaluated as the baseline model. A content-based fallback will be used for cold-start students.


## Week 6 – Model Experiments and Reproducibility

### 1. Environment Setup

Requirements:
- Python 3.12
- pandas
- scikit-learn
- scikit-surprise
- MLflow
- numpy
- joblib

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Model Experiments

Run the following commands from the project root:

```bash
python src/train_cf.py
python src/tune_svd.py
python src/compare_svd_ndcg.py
python src/compare_models.py
python src/evaluate_hybrid.py
python src/tune_hybrid.py
python src/hybrid_recommender.py
```

### 3. MLflow Experiment Tracking

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

View experiment runs, parameters, metrics, and logged artifacts.

### 4. Evaluation Reports

Generated reports are available in the `reports/` directory:

- svd_experiment_results.csv
- svd_tuning_results.csv
- svd_ndcg_comparison.csv
- fair_model_comparison.csv
- content_based_evaluation.csv
- hybrid_evaluation.csv
- hybrid_weight_tuning.csv

### 5. Evaluation Summary

The chronological model comparison produced:

| Model | NDCG@10 |
|---|---:|
| Popularity | 0.1207 |
| SVD | 0.0561 |
| Content-Based | 0.1083 |
| Hybrid | 0.0806 |

The target NDCG@10 of 0.55 has not yet been achieved. Further model improvements and evaluation are required.

### 6. GitHub Repository

https://github.com/ameeshashaik279/edutrack-ai-learning-recommender

### MLflow Experiment Tracking

Experiment Name: EduTrack_SVD_Experiments

| Run | n_factors | n_epochs | reg_all | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| SVD_Config_1 | 20 | 20 | 0.02 | 1.0602 | 0.9238 |
| SVD_Config_2 | 50 | 30 | 0.05 | 1.0699 | 0.9299 |

Both experiments completed successfully and were tracked using MLflow.

Best configuration based on RMSE and MAE: SVD_Config_1.

### Reproducing MLflow Experiments

1. Install project dependencies:

   pip install -r requirements.txt

2. Run the SVD experiment:

   python src/train_cf.py

3. Start the MLflow UI:

   mlflow ui --backend-store-uri sqlite:///mlflow.db

4. Open the MLflow dashboard:

   http://127.0.0.1:5000

5. Select EduTrack_SVD_Experiments to compare the experiment runs.

Note: Run the commands from the project root directory.