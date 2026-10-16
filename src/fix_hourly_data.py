
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "traffic_cleaned.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "traffic_hourly.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date_time"])
df = df.sort_values("date_time")

# Aggregate duplicate timestamps:
# Average numeric observations; keep the first categorical observation.
numeric_cols = [
    "temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"
]
categorical_cols = [
    "holiday", "weather_main", "weather_description"
]

aggregations = {
    col: "mean" for col in numeric_cols if col in df.columns
}
aggregations.update({
    col: "first" for col in categorical_cols if col in df.columns
})

df = df.groupby("date_time", as_index=False).agg(aggregations)
df = df.set_index("date_time").sort_index()

# Create a complete hourly timeline.
full_index = pd.date_range(
    start=df.index.min().floor("h"),
    end=df.index.max().floor("h"),
    freq="h"
)
df = df.reindex(full_index)
df.index.name = "date_time"

# Recompute time features from the timestamp.
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# Keep inserted hours missing; do not invent traffic measurements.
df = df.reset_index()
df.to_csv(OUTPUT_FILE, index=False)

print("Hourly dataset saved:", OUTPUT_FILE)
print("Rows:", len(df))
print("Duplicate timestamps:", df["date_time"].duplicated().sum())
print("Missing traffic measurements:", df["traffic_volume"].isna().sum())
print("Missing hours inserted:", df["traffic_volume"].isna().sum())

intervals = df["date_time"].diff().dropna()
print("Intervals not exactly one hour:",
      int((intervals != pd.Timedelta(hours=1)).sum()))
print("\nMissing values by column:")
print(df.isna().sum())
