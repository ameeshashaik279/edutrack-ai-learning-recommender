

import pandas as pd
from pathlib import Path


def clean_data(df):
    """Clean and validate a dataset."""
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.drop_duplicates()
    return df


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "aiml_student_performance.csv",
        "aiml_content_engagement.csv",
    ]

    for filename in files:
        print(f"\nProcessing {filename}...")

        df = pd.read_csv(raw_dir / filename)
        df = clean_data(df)

        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)

        print("Shape:", df.shape)
        print("Missing values:", df.isnull().sum().sum())
        print("Saved to:", output_path)
