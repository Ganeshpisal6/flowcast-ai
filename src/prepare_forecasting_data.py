
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "traffic_cleaned.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "traffic_forecasting.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# Forecast horizons for hourly data
for hours_ahead in [1, 2, 3, 4]:
    df[f"traffic_next_{hours_ahead}h"] = (
        df["traffic_volume"].shift(-hours_ahead)
    )

# Previous traffic observations available at prediction time
for lag in [1, 2, 3, 4, 24]:
    df[f"traffic_lag_{lag}h"] = df["traffic_volume"].shift(lag)

# Rolling averages use past observations only
df["traffic_mean_previous_3h"] = (
    df["traffic_volume"].shift(1).rolling(3).mean()
)
df["traffic_mean_previous_24h"] = (
    df["traffic_volume"].shift(1).rolling(24).mean()
)

# Remove rows without the required historical or future values
required_columns = [
    "traffic_next_1h",
    "traffic_next_2h",
    "traffic_next_3h",
    "traffic_next_4h",
    "traffic_lag_1h",
    "traffic_lag_2h",
    "traffic_lag_3h",
    "traffic_lag_4h",
    "traffic_lag_24h",
    "traffic_mean_previous_3h",
    "traffic_mean_previous_24h",
]

df = df.dropna(subset=required_columns).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Forecasting dataset created successfully.")
print("Shape:", df.shape)
print("Saved to:", OUTPUT_FILE)
print("\nForecast targets:")
print([
    "traffic_next_1h",
    "traffic_next_2h",
    "traffic_next_3h",
    "traffic_next_4h",
])
print("\nMissing values in forecast targets:")
print(df[[
    "traffic_next_1h",
    "traffic_next_2h",
    "traffic_next_3h",
    "traffic_next_4h",
]].isna().sum())
