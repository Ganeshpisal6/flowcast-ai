import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA = Path("data/processed/pems04_congestion_classification.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

features = ["flow", "occupancy", "sensor_id", "interval_index"]
target = "congestion_level"

df = df.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)

train_parts = []
test_parts = []

for sensor_id, group in df.groupby("sensor_id", sort=False):
    split_index = int(len(group) * 0.8)
    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )
}

metrics = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Free-flow", "Moderate", "Heavy", "Severe"]
    ))

    joblib.dump(model, MODEL_DIR / f"pems04_{name}_congestion.joblib")

    metrics.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_rows": len(train_df),
        "test_rows": len(test_df)
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(REPORT_DIR / "pems04_congestion_metrics.csv", index=False)

print("\nFinal comparison:")
print(metrics_df.to_string(index=False))
print("\nMetrics saved to reports/pems04_congestion_metrics.csv")
