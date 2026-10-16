
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
REPORT_FILE = ROOT / "reports" / "model_comparison.csv"

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

required_columns = features + targets
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in dataset: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()
df = df.sort_values("date_time").reset_index(drop=True)

if len(df) < 10:
    raise ValueError("Not enough valid rows to train and evaluate models.")

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
X_test = test[features]

models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for target in targets:
    print(f"\n{'=' * 60}")
    print(f"Forecast target: {target}")
    print("=" * 60)

    y_train = train[target]
    y_test = test[target]

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Target": target,
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        })

        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["Target", "MAE"]
).reset_index(drop=True)

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_FILE, index=False)

print("\nMODEL COMPARISON SUMMARY")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {REPORT_FILE}")
