
import pandas as pd


def train_model(engagements):
    required = {
        "student_id",
        "subject",
        "content_type",
        "engagement_score",
    }

    missing = required - set(engagements.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = engagements.copy()
    data["engagement_score"] = pd.to_numeric(
        data["engagement_score"], errors="coerce"
    )
    data = data.dropna(subset=list(required))

    if data.empty:
        raise ValueError("No valid engagement data found.")

    scores = (
        data.groupby(["subject", "content_type"])["engagement_score"]
        .mean()
        .sort_values(ascending=False)
    )

    seen = data.groupby("student_id")["subject"].apply(set).to_dict()

    return {
        "scores": scores,
        "seen": seen,
    }


def recommend_courses(model, student_id, n=5):
    seen_subjects = model["seen"].get(student_id, set())
    recommendations = []

    for (subject, content_type), score in model["scores"].items():
        if subject in seen_subjects:
            continue

        recommendations.append({
            "subject": subject,
            "content_type": content_type,
            "score": round(float(score), 3),
        })

        if len(recommendations) >= n:
            break

    return recommendations


