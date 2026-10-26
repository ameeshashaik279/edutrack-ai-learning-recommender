
import pandas as pd


def load_student_data(path):
    """Load student performance data."""
    return pd.read_csv(path)


def load_engagement_data(path):
    """Load content engagement data."""
    return pd.read_csv(path)


def clean_data(df):
    """Clean column names, duplicates, and missing values."""
    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Report missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    print("EduTrack data pipeline initialized.")
