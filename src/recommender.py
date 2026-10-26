
import pandas as pd


def train_model(engagements):
    """Build a subject and content-type recommendation baseline."""

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

    # Average engagement for each subject and content type
    content_scores = (
        data.groupby(["subject", "content_type"])["engagement_score"]
        .mean()
        .sort_values(ascending=False)
    )

    # Track subject and content types already seen by each student
    seen = (
        data.groupby("student_id")
        .apply(
            lambda group: set(
                zip(group["subject"], group["content_type"])
            )
        )
        .to_dict()
    )

    return {
        "content_scores": content_scores,
        "seen": seen,
    }


def recommend_courses(model, student_id, n=5):
    """Recommend unseen subject and content-type combinations."""

    if n <= 0:
        return []

    seen_items = model["seen"].get(student_id, set())
    recommendations = []

    for (subject, content_type), score in model["content_scores"].items():
        if (subject, content_type) in seen_items:
            continue

        recommendations.append({
            "subject": subject,
            "content_type": content_type,
            "score": round(float(score), 3),
        })

        if len(recommendations) >= n:
            break

    return recommendations
