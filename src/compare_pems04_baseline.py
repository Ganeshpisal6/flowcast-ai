from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/processed/pems04_30min_forecasting.csv")
REPORT = Path("reports/pems04_flow_30min_model_comparison.csv")
TARGET = "flow_next_30min"
FEATURES = ["flow", "occupancy", "speed"]

df = pd.read_csv(DATA)

# Match the training script's chronological 80/20 split within each sensor.
df = df.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)
train_parts = []
test_parts = []

for _, group in df.groupby("sensor_id", sort=False):
    split = int(len(group) * 0.8)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

test = pd.concat(test_parts, ignore_index=True)
y_true = test[TARGET].to_numpy()

models = {
    "persistence_baseline": None,
    "random_forest": Path("models/pems04_random_forest_flow_30min.joblib"),
    "hist_gradient_boosting": Path("models/pems04_hist_gradient_boosting_flow_30min.joblib"),
}

results = []

for name, model_path in models.items():
    if name == "persistence_baseline":
        # Predict next 30-minute flow using current flow.
        predictions = test["flow"].to_numpy()
    else:
        model = joblib.load(model_path)
        predictions = model.predict(test[FEATURES])

    results.append({
        "model": name,
        "target": TARGET,
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
        "test_rows": len(test),
    })

comparison = pd.DataFrame(results).sort_values("RMSE")
REPORT.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(REPORT, index=False)

print("\n30-minute flow forecast comparison (lower MAE/RMSE is better; higher R2 is better)")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved report: {REPORT.resolve()}")
