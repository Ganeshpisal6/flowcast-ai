
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "traffic_hourly.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# Create true future targets on the continuous hourly timeline.
for h in [1, 2, 3, 4]:
    df[f"traffic_next_{h}h"] = df["traffic_volume"].shift(-h)

# Historical traffic features available at prediction time.
for lag in [1, 2, 3, 4, 24]:
    df[f"traffic_lag_{lag}h"] = df["traffic_volume"].shift(lag)

df["traffic_mean_previous_3h"] = (
    df["traffic_volume"].shift(1).rolling(3, min_periods=3).mean()
)
df["traffic_mean_previous_24h"] = (
    df["traffic_volume"].shift(1).rolling(24, min_periods=24).mean()
)

feature_columns = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend",
    "traffic_lag_1h", "traffic_lag_2h", "traffic_lag_3h",
    "traffic_lag_4h", "traffic_lag_24h",
    "traffic_mean_previous_3h", "traffic_mean_previous_24h",
]

target_columns = [
    "traffic_next_1h", "traffic_next_2h",
    "traffic_next_3h", "traffic_next_4h",
]

# Keep only complete examples. Do not fabricate missing observations.
before = len(df)
df = df.dropna(subset=feature_columns + target_columns).copy()
after = len(df)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Forecasting dataset v2 created.")
print("Rows before filtering:", before)
print("Valid forecasting rows:", after)
print("Rows excluded due to missing inputs/targets:", before - after)
print("Saved to:", OUTPUT_FILE)
print("\nMissing values in model features:")
print(df[feature_columns].isna().sum().sum())
print("Missing values in future targets:")
print(df[target_columns].isna().sum().sum())
print("\nForecast target columns:", target_columns)
