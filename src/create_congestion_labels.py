import pandas as pd
from pathlib import Path

input_file = Path("data/processed/pems04_30min_forecasting.csv")
output_file = Path("data/processed/pems04_congestion_classification.csv")

df = pd.read_csv(input_file)

def classify_congestion(speed):
    if speed >= 60:
        return "Free-flow"
    elif speed >= 50:
        return "Moderate"
    elif speed >= 40:
        return "Heavy"
    else:
        return "Severe"

df["congestion_level"] = df["speed"].apply(classify_congestion)

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("Congestion dataset created successfully.")
print("Shape:", df.shape)

print("\nClass distribution:")
print(df["congestion_level"].value_counts())

print("\nClass percentages:")
print((df["congestion_level"].value_counts(normalize=True) * 100).round(2))

print("\nMissing labels:", df["congestion_level"].isna().sum())
print("Saved to:", output_file)
