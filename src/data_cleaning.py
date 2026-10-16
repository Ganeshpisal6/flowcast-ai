
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_traffic"
    / "Metro_Interstate_Traffic_Volume.csv.gz"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "traffic_cleaned.csv"
)


def clean_traffic_data():
    """Load, validate, clean, and save traffic data."""

    print("Loading raw traffic dataset...")
    df = pd.read_csv(RAW_FILE)

    print(f"Original shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column to datetime
    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    # Remove rows without a valid date or target value
    df = df.dropna(subset=["date_time", "traffic_volume"])

    # Remove duplicate records
    df = df.drop_duplicates()

    # Sort records chronologically
    df = df.sort_values("date_time").reset_index(drop=True)

    # Fill missing holiday values
    df["holiday"] = df["holiday"].fillna("No Holiday")

    # Fill missing weather categories
    for column in ["weather_main", "weather_description"]:
        df[column] = df[column].fillna("Unknown")

    # Fill missing numeric weather values with their medians
    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Create time-based prediction features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Assign congestion classes using traffic volume.
    # These are initial project thresholds, not official road standards.
    df["congestion_level"] = pd.cut(
        df["traffic_volume"],
        bins=[-1, 1000, 3000, 5000, float("inf")],
        labels=[
            "Free-flow",
            "Moderate",
            "Heavy",
            "Severe",
        ],
    )

    # Save cleaned dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    print("\nCongestion class counts:")
    print(df["congestion_level"].value_counts())


if __name__ == "__main__":
    clean_traffic_data()
