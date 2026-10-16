
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

features = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend",
]

targets = [
    "traffic_next_1h",
    "traffic_next_2h",
    "traffic_next_3h",
    "traffic_next_4h",
]

required = features + targets
df = df.dropna(subset=required).reset_index(drop=True)

split = int(len(df) * 0.8)
gap = 4

train = df.iloc[:split - gap]
test = df.iloc[split:]

if train.empty or test.empty:
    raise ValueError("Not enough data for the selected train/test split.")

X_train = train[features]
X_test = test[features]

results = []

for target in targets:
    print(f"\nTraining model for {target}...")

    y_train = train[target]
    y_test = test[target]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    model_file = MODEL_DIR / f"xgboost_{target}.joblib"

    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
        },
        model_file,
    )

    results.append({
        "Target": target,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Model": "XGBoost",
    })

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    print(f"Saved: {model_file}")

results_df = pd.DataFrame(results)
report_file = REPORT_DIR / "forecast_model_metrics.csv"
results_df.to_csv(report_file, index=False)

print("\nFINAL FORECAST MODEL METRICS")
print(results_df.to_string(index=False))
print(f"\nMetrics saved to: {report_file}")
