
from pathlib import Path
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
MODEL_DIR = ROOT / "models"
REPORT_FILE = ROOT / "reports" / "forecast_horizon_comparison.csv"

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

results = []

for horizon in [1, 2, 3, 4]:
    target = f"traffic_next_{horizon}h"
    print(f"\n{'=' * 45}")
    print(f"Training forecast: {horizon} hour(s) ahead")

  
    # Split chronologically and prevent future-target overlap.
    split = int(len(df) * 0.8)
    test_start_time = df.iloc[split]["date_time"]

    train = df.loc[
        df["date_time"] + pd.Timedelta(hours=horizon) < test_start_time
    ]
    test = df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]



    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

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
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"XGBoost MAE:  {mae:.2f}")
    print(f"XGBoost RMSE: {rmse:.2f}")
    print(f"XGBoost R2:   {r2:.4f}")

    results.append({
        "Horizon_hours": horizon,
        "Baseline_MAE": round(baseline_mae, 2),
        "XGBoost_MAE": round(mae, 2),
        "XGBoost_RMSE": round(rmse, 2),
        "XGBoost_R2": round(r2, 4),
    })

    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "horizon_hours": horizon,
        },
        MODEL_DIR / f"xgboost_traffic_{horizon}h.joblib",
    )

results_df = pd.DataFrame(results)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_FILE, index=False)

print("\nFORECAST HORIZON COMPARISON")
print(results_df.to_string(index=False))
print("\nResults saved to:", REPORT_FILE)
print("Models saved in:", MODEL_DIR)
