import pandas as pd


def load_student_data(path):
    """Load student performance data."""
    return pd.read_csv(path)


def load_engagement_data(path):
    """Load content engagement data."""
    return pd.read_csv(path)


def clean_data(df):
    """Basic data cleaning."""
    df = df.copy()
    df = df.drop_duplicates()
    return df


if __name__ == "__main__":
    print("EduTrack data pipeline initialized.")
