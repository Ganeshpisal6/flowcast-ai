from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "pems04_30min_forecasting.csv.gz"
REPORT_DIR = ROOT / "reports"
OUTPUT_FILE = REPORT_DIR / "pems04_persistence_baseline_comparison.csv"

FEATURES = ["flow", "speed"]
HORIZONS = [30, 60, 90, 120]

data = pd.read_csv(DATA_FILE)
data = data.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)

data["row_number"] = data.groupby("sensor_id").cumcount()
data["sensor_rows"] = data.groupby("sensor_id")["interval_index"].transform("count")
test = data[data["row_number"] >= data["sensor_rows"] * 0.8].copy()

results = []

for variable in FEATURES:
    for horizon in HORIZONS:
        target = f"{variable}_next_{horizon}min"
        actual = test[target].to_numpy()
        predicted = test[variable].to_numpy()

        results.append({
            "model": "persistence_baseline",
            "variable": variable,
            "target": target,
            "horizon_minutes": horizon,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
            "test_rows": len(test),
        })

baseline = pd.DataFrame(results)

flow_metrics = pd.read_csv(
    ROOT / "reports" / "pems04_flow_all_horizons_metrics.csv"
)
speed_metrics = pd.read_csv(
    ROOT / "reports" / "pems04_speed_all_horizons_metrics.csv"
)

models = pd.concat([flow_metrics, speed_metrics], ignore_index=True)
models["variable"] = models["target"].str.split("_").str[0]
models["horizon_minutes"] = models["target"].str.extract(r"(\d+)min").astype(int)

models = models[[
    "model", "variable", "target", "horizon_minutes",
    "MAE", "RMSE", "R2", "test_rows"
]]

comparison = pd.concat([models, baseline], ignore_index=True)
comparison = comparison.sort_values(
    ["variable", "horizon_minutes", "RMSE"]
).reset_index(drop=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(OUTPUT_FILE, index=False)

print("\n========== FOUR-HORIZON MODEL COMPARISON ==========")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nSaved report: {OUTPUT_FILE}")
