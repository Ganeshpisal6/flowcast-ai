from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pems04_congestion_classification.csv.gz"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = ["flow", "occupancy", "sensor_id", "interval_index"]
TARGET = "congestion_level"
LABELS = ["Free-flow", "Moderate", "Heavy", "Severe"]

print("Loading PEMS04 congestion data...", flush=True)
df = pd.read_csv(DATA)
df = df.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)

df["row_number"] = df.groupby("sensor_id").cumcount()
df["sensor_rows"] = df.groupby("sensor_id")["interval_index"].transform("count")

train = df[df["row_number"] < df["sensor_rows"] * 0.8].copy()
test = df[df["row_number"] >= df["sensor_rows"] * 0.8].copy()

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

sample_size = min(160000, len(train))

X_sample, _, y_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=sample_size,
    stratify=y_train,
    random_state=42
)

print(f"Training rows sampled: {len(X_sample)}", flush=True)
print(f"Chronological test rows: {len(X_test)}", flush=True)

model = make_pipeline(
    StandardScaler(),
    LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
        dual="auto"
    )
)

print("\nTraining Linear SVM...", flush=True)
model.fit(X_sample, y_sample)

print("Evaluating model...", flush=True)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro"
)

print("\n========== LINEAR SVM RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=LABELS,
    zero_division=0
))

print("Confusion matrix:")
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=LABELS
)
print(matrix)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "pems04_linear_svm_congestion.joblib"
)

pd.DataFrame([{
    "model": "linear_svm",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "train_rows": len(X_sample),
    "test_rows": len(X_test),
    "training_method": "stratified_sample"
}]).to_csv(
    REPORT_DIR / "pems04_svm_congestion_metrics.csv",
    index=False
)

pd.DataFrame(
    matrix,
    index=LABELS,
    columns=LABELS
).to_csv(
    REPORT_DIR / "pems04_svm_confusion_matrix.csv"
)

print("\nSaved model and both evaluation reports.")
