from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "pems04_30min_forecasting.csv.gz"
REPORT = ROOT / "reports" / "pems04_numpy_linear_regression_metrics.csv"

print("Loading PEMS04 data...")
df = pd.read_csv(DATA)
features = ["flow", "occupancy", "speed"]
target = "flow_next_30min"

df = df.dropna(subset=features + [target]).copy()
df = df.sort_values(["sensor_id", "interval_index"])

# Chronological 80/20 split separately for every sensor
df["position"] = df.groupby("sensor_id").cumcount()
df["sensor_count"] = df.groupby("sensor_id")["position"].transform("max") + 1
df["train_row"] = df["position"] < (df["sensor_count"] * 0.8)

train = df[df["train_row"]]
test = df[~df["train_row"]]

# Reproducible training sample
sample_size = min(160000, len(train))
train = train.sample(n=sample_size, random_state=42)

X_train = train[features].to_numpy(dtype=np.float64)
y_train = train[target].to_numpy(dtype=np.float64)
X_test = test[features].to_numpy(dtype=np.float64)
y_test = test[target].to_numpy(dtype=np.float64)

# Standardize using training data only
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
x_std[x_std == 0] = 1

X_train = (X_train - x_mean) / x_std
X_test = (X_test - x_mean) / x_std

y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std

# Add intercept column
X_train = np.column_stack([np.ones(len(X_train)), X_train])
X_test = np.column_stack([np.ones(len(X_test)), X_test])

# NumPy batch gradient descent
weights = np.zeros(X_train.shape[1])
learning_rate = 0.05
epochs = 500

print("Training NumPy Linear Regression with gradient descent...")
for epoch in range(epochs):
    predictions = X_train @ weights
    errors = predictions - y_train_scaled
    gradients = (2 / len(X_train)) * (X_train.T @ errors)
    weights -= learning_rate * gradients

# Predict in original flow units
predictions = (X_test @ weights) * y_std + y_mean

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
ss_total = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - np.sum((y_test - predictions) ** 2) / ss_total

result = pd.DataFrame([{
    "model": "NumPy Linear Regression (Gradient Descent)",
    "target": target,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "epochs": epochs,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

REPORT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(REPORT, index=False)

print("\n========== NUMPY LINEAR REGRESSION RESULTS ==========")
print(result.to_string(index=False))
print(f"\nSaved report: {REPORT}")
