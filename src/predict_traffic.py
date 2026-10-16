from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend",
    "traffic_lag_1h", "traffic_lag_2h", "traffic_lag_3h",
    "traffic_lag_4h", "traffic_lag_24h",
    "traffic_mean_previous_3h", "traffic_mean_previous_24h",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Convert features to numeric and find the latest valid row.
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_rows = df.dropna(subset=FEATURES)

    if valid_rows.empty:
        raise ValueError("No rows have all required numeric input features.")

    latest = valid_rows.iloc[-1]
    X = latest[FEATURES].to_frame().T.astype("float64")

    print("\n" + "=" * 55)
    print("          FLOWCAST AI - TRAFFIC FORECAST")
    print("=" * 55)
    print(f"Latest input timestamp: {latest['date_time']}")
    print("\nPredicted traffic for future horizons:\n")

    for horizon in [1, 2, 3, 4]:
        model_path = MODEL_DIR / f"xgboost_traffic_{horizon}h.joblib"

        if not model_path.exists():
            print(f"{horizon}-hour model not found: {model_path.name}")
            continue

        bundle = joblib.load(model_path)
        model = bundle["model"]
        model_features = bundle["features"]

        prediction = model.predict(X[model_features])[0]
        forecast_time = latest["date_time"] + pd.Timedelta(hours=horizon)

        print(f"Forecast horizon : {horizon} hour(s)")
        print(f"Forecast time    : {forecast_time}")
        print(f"Predicted traffic: {prediction:,.2f}")
        print("-" * 55)

    print("\nForecast generation completed.")


if __name__ == "__main__":
    main()
