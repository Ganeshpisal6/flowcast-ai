
from pathlib import Path
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

features = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend",
    "traffic_lag_1h", "traffic_lag_2h", "traffic_lag_3h",
    "traffic_lag_4h", "traffic_lag_24h",
    "traffic_mean_previous_3h", "traffic_mean_previous_24h",
]

target = "traffic_next_1h"

# Chronological split: earlier records train the model;
# later records evaluate it.
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Forecast horizon: 1 hour ahead")

# Baseline for comparison
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# Forecasting model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

def show_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

baseline_metrics = show_metrics(
    "Mean Baseline", y_test, baseline_predictions
)
model_metrics = show_metrics(
    "XGBoost - 1 Hour Ahead", y_test, predictions
)

model_path = MODEL_DIR / "xgboost_traffic_1h.joblib"

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": target,
        "horizon_hours": 1,
        "metrics": model_metrics,
    },
    model_path,
)

print("\nTop feature importances:")
print(
    pd.Series(model.feature_importances_, index=features)
    .sort_values(ascending=False)
    .head(10)
    .round(4)
)

print("\nSaved forecasting model to:", model_path)
