
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "pems04_30min_forecasting.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = ["flow", "occupancy", "speed"]
TARGET = "flow_next_30min"


def main():
    data = pd.read_csv(DATA_FILE).sort_values(
        ["sensor_id", "interval_index"]
    )

    # A time-ordered split for every sensor: first 80% train, last 20% test.
    data["row_number"] = data.groupby("sensor_id").cumcount()
    data["sensor_rows"] = data.groupby("sensor_id")["interval_index"].transform("count")
    data["train_row"] = data["row_number"] < (data["sensor_rows"] * 0.8)

    train = data[data["train_row"]]
    test = data[~data["train_row"]]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    results = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.4f}")

        results.append({
            "model": name,
            "target": TARGET,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "train_rows": len(train),
            "test_rows": len(test),
        })

        joblib.dump(model, MODEL_DIR / f"pems04_{name}_flow_30min.joblib")

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "pems04_flow_30min_model_metrics.csv",
        index=False,
    )
    print("\nSaved model metrics to reports/pems04_flow_30min_model_metrics.csv")


if __name__ == "__main__":
    main()
