
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
    student_path = "data/raw/aiml_student_performance.csv"
    engagement_path = "data/raw/aiml_content_engagement.csv"

    print("Loading student performance data...")
    student_df = clean_data(load_student_data(student_path))
    print("Student dataset shape:", student_df.shape)

    print("\nLoading content engagement data...")
    engagement_df = clean_data(load_engagement_data(engagement_path))
    print("Engagement dataset shape:", engagement_df.shape)


