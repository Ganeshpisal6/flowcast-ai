
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "processed" / "traffic_cleaned.csv"

df = pd.read_csv(FILE, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

gaps = df["date_time"].diff()
gap_counts = gaps.value_counts().sort_index()

print("Total records:", len(df))
print("Start:", df["date_time"].min())
print("End:", df["date_time"].max())
print("\nMost common timestamp intervals:")
print(gap_counts.head(10))

print("\nIntervals other than one hour:")
irregular = gaps[gaps != pd.Timedelta(hours=1)].dropna()
print("Count:", len(irregular))
print(irregular.value_counts().head(10))

print("\nDuplicate timestamps:", df["date_time"].duplicated().sum())
