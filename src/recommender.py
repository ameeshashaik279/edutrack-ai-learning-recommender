

import pandas as pd


def train_model(engagements):
    """Build a simple baseline from content engagement scores."""

    required = {"student_id", "content_type", "subject", "engagement_score"}
    missing = required - set(engagements.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = engagements.dropna(
        subset=["student_id", "content_type", "subject", "engagement_score"]
    ).copy()

    if data.empty:
        raise ValueError("No valid engagement data available.")

    data["engagement_score"] = pd.to_numeric(
        data["engagement_score"], errors="coerce"
    )
    data = data.dropna(subset=["engagement_score"])

    if data.empty:
        raise ValueError("No numeric engagement scores available.")

    model = {
        "data": data,
        "content_scores": (
            data.groupby(["subject", "content_type"])["engagement_score"]
            .mean()
            .sort_values(ascending=False)
        ),
        "seen": data.groupby("student_id")["subject"].apply(set).to_dict(),
    }

    return model


def recommend_courses(model, student_id, n=5):
    """Suggest subjects the student has not engaged with yet."""

    seen_subjects = model["seen"].get(student_id, set())
    scores = model["content_scores"]

    recommendations = []

    for (subject, content_type), score in scores.items():
        if subject not in seen_subjects:
            recommendations.append({
                "subject": subject,
                "content_type": content_type,
                "score": round(float(score), 3),
            })

        if len(recommendations) >= max(0, n):
            break

    return recommendations


if __name__ == "__main__":
    print("EduTrack engagement baseline initialized.")
