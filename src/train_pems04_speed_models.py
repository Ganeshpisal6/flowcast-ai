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
HORIZONS = [30, 60, 90, 120]


def make_models():
    return {
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


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE).sort_values(
        ["sensor_id", "interval_index"]
    ).reset_index(drop=True)

    data["row_number"] = data.groupby("sensor_id").cumcount()
    data["sensor_rows"] = data.groupby("sensor_id")["interval_index"].transform("count")
    data["train_row"] = data["row_number"] < (data["sensor_rows"] * 0.8)

    train = data[data["train_row"]]
    test = data[~data["train_row"]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for horizon in HORIZONS:
        target = f"speed_next_{horizon}min"
        if target not in data.columns:
            raise ValueError(f"Missing target column: {target}")

        print(f"\n{'=' * 55}\nTarget: {target}\n{'=' * 55}", flush=True)

        X_train = train[FEATURES]
        y_train = train[target]
        X_test = test[FEATURES]
        y_test = test[target]

        for name, model in make_models().items():
            print(f"\nTraining {name}...", flush=True)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            print(f"MAE:  {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"R2:   {r2:.4f}")

            model_path = MODEL_DIR / f"pems04_{name}_speed_{horizon}min.joblib"
            joblib.dump(model, model_path)

            results.append({
                "model": name,
                "target": target,
                "horizon_minutes": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "train_rows": len(train),
                "test_rows": len(test),
            })

    metrics_file = REPORT_DIR / "pems04_speed_all_horizons_metrics.csv"
    pd.DataFrame(results).to_csv(metrics_file, index=False)

    print(f"\nAll speed-horizon metrics saved to: {metrics_file}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
