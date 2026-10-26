

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

    """Recommend unseen courses using the baseline mean rating."""
    seen = model.seen_courses.get(student_id, set())

    candidates = [
        course_id
        for course_id in model.course_ids
        if course_id not in seen
    ]

    return [
        {
            "course_id": course_id,
            "score": model.mean_rating
        }
        for course_id in candidates[:max(0, n)]
    ]


if __name__ == "__main__":
    print("EduTrack baseline recommender initialized.")
 223dfe85df6edc88c79a0f6f82b3fa421b1156b8
