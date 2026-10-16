
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "traffic_cleaned.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])

print("DATASET OVERVIEW")
print("-" * 40)
print("Shape:", df.shape)
print("Date range:", df["date_time"].min(), "to", df["date_time"].max())
print("\nDescriptive statistics:")
print(df[["traffic_volume", "temp", "rain_1h", "clouds_all"]].describe())

# 1. Average traffic by hour
hourly_traffic = df.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
hourly_traffic.plot(kind="line", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "traffic_by_hour.png")
plt.close()

# 2. Average traffic by day of week
daily_traffic = df.groupby("day_of_week")["traffic_volume"].mean()
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
daily_traffic.index = [day_names[i] for i in daily_traffic.index]

plt.figure(figsize=(10, 5))
daily_traffic.plot(kind="bar")
plt.title("Average Traffic Volume by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_DIR / "traffic_by_day.png")
plt.close()

# 3. Distribution of traffic volume
plt.figure(figsize=(10, 5))
df["traffic_volume"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(REPORT_DIR / "traffic_distribution.png")
plt.close()

print("\nCongestion class distribution:")
print(df["congestion_level"].value_counts())

print("\nEDA completed.")
print("Charts saved in:", REPORT_DIR)
