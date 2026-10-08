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
