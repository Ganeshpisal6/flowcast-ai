from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "pems04_30min_forecasting.csv.gz"
REPORT_FILE = ROOT / "reports" / "pems04_classical_model_comparison.csv"

FEATURES = ["flow", "occupancy", "speed"]
TARGETS = ["flow_next_30min", "speed_next_30min"]

df = pd.read_csv(DATA_FILE)
df = df.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)

df["row_number"] = df.groupby("sensor_id").cumcount()
df["sensor_rows"] = df.groupby("sensor_id")["interval_index"].transform("count")

train = df[df["row_number"] < df["sensor_rows"] * 0.8].copy()
test = df[df["row_number"] >= df["sensor_rows"] * 0.8].copy()

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=10,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )
}

results = []

for target in TARGETS:
    print(f"\n========== {target} ==========", flush=True)

    valid_train = train.dropna(subset=FEATURES + [target])
    valid_test = test.dropna(subset=FEATURES + [target])

    X_train = valid_train[FEATURES]
    y_train = valid_train[target]
    X_test = valid_test[FEATURES]
    y_test = valid_test[target]

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "train_rows": len(valid_train),
            "test_rows": len(valid_test)
        })

        print(
            f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}",
            flush=True
        )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["target", "RMSE"])

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_FILE, index=False)

print("\n========== CLASSICAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nReport saved: {REPORT_FILE}")
