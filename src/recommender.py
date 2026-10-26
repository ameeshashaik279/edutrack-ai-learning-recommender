
import pandas as pd
from sklearn.dummy import DummyRegressor


def train_model(interactions):
    """
    Train a simple baseline model using the mean rating.
    Required columns: student_id, course_id, rating.
    """
    required = {"student_id", "course_id", "rating"}
    missing = required - set(interactions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = interactions.dropna(
        subset=["student_id", "course_id", "rating"]
    ).copy()

    if data.empty:
        raise ValueError("No valid interaction data available.")

    model = DummyRegressor(strategy="mean")
    model.fit(
        [[0] for _ in range(len(data))],
        data["rating"]
    )

    model.mean_rating = float(data["rating"].mean())
    model.course_ids = data["course_id"].drop_duplicates().tolist()
    model.seen_courses = (
        data.groupby("student_id")["course_id"]
        .apply(set)
        .to_dict()
    )

    return model


def recommend_courses(model, student_id, n=5):
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
