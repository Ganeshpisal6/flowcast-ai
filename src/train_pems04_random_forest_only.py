from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["flow", "occupancy", "speed"]
CONGESTION_FEATURES = ["flow", "occupancy", "sensor_id", "interval_index"]
HORIZONS = [30, 60, 90, 120]

def make_regressor():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )

start = time.time()

# FLOW AND SPEED FORECASTING
data = pd.read_csv(DATA_DIR / "pems04_30min_forecasting.csv.gz")
data = data.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)

data["row_number"] = data.groupby("sensor_id").cumcount()
data["sensor_rows"] = data.groupby("sensor_id")["interval_index"].transform("count")
train_mask = data["row_number"] < (data["sensor_rows"] * 0.8)

train = data.loc[train_mask]
test = data.loc[~train_mask]

for target_type in ["flow", "speed"]:
    for horizon in HORIZONS:
        target = f"{target_type}_next_{horizon}min"

        print(f"\nTraining Random Forest: {target}", flush=True)

        model = make_regressor()
        model.fit(train[FEATURES], train[target])

        predictions = model.predict(test[FEATURES])

        mae = mean_absolute_error(test[target], predictions)
        rmse = np.sqrt(mean_squared_error(test[target], predictions))
        r2 = r2_score(test[target], predictions)

        filename = f"pems04_random_forest_{target_type}_{horizon}min.joblib"
        joblib.dump(model, MODEL_DIR / filename)

        print(f"Saved: {filename}", flush=True)
        print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}", flush=True)

# CONGESTION CLASSIFICATION
print("\nTraining Random Forest congestion classifier...", flush=True)

congestion = pd.read_csv(
    DATA_DIR / "pems04_congestion_classification.csv.gz"
)
congestion = congestion.sort_values(
    ["sensor_id", "interval_index"]
).reset_index(drop=True)

train_parts = []
test_parts = []

for _, group in congestion.groupby("sensor_id", sort=False):
    split = int(len(group) * 0.8)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

classifier.fit(
    train_df[CONGESTION_FEATURES],
    train_df["congestion_level"],
)

predictions = classifier.predict(test_df[CONGESTION_FEATURES])

accuracy = accuracy_score(test_df["congestion_level"], predictions)
macro_f1 = f1_score(
    test_df["congestion_level"], predictions, average="macro"
)

filename = "pems04_random_forest_congestion.joblib"
joblib.dump(classifier, MODEL_DIR / filename)

print(f"Saved: {filename}", flush=True)
print(f"Accuracy={accuracy:.4f} | Macro-F1={macro_f1:.4f}", flush=True)

print("\n========== TRAINING COMPLETE ==========")
print(f"Elapsed minutes: {(time.time() - start) / 60:.1f}")
print("Random Forest models saved in:", MODEL_DIR)
