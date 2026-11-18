# FlowCast AI - Model Documentation

## 1. Project Overview

FlowCast AI uses machine learning to forecast traffic flow and average speed using the PEMS04 multi-sensor dataset.

The dataset contains 307 sensors. Five-minute observations are aggregated into 30-minute intervals.

Forecast horizons:
- 30 minutes
- 60 minutes
- 90 minutes
- 120 minutes

## 2. Regression Models

The following algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Hist Gradient Boosting Regressor
- XGBoost Regressor
- NumPy-based Linear Regression using gradient descent

Classical model comparisons include Linear Regression, Decision Tree and XGBoost for 30-minute targets. Random Forest and Hist Gradient Boosting were evaluated across all four forecasting horizons.

## 3. Classification Models

Congestion classification uses four speed-based categories:

- Free-flow: speed >= 60
- Moderate: speed >= 50 and < 60
- Heavy: speed >= 40 and < 50
- Severe: speed < 40

Evaluated classifiers:
- Random Forest Classifier
- Hist Gradient Boosting Classifier
- Linear Support Vector Machine

These labels are proxy labels derived from speed thresholds and are not independently verified congestion records.

## 4. Training and Evaluation

The PEMS04 experiments use chronological, per-sensor training and testing splits.

Regression metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

Classification metrics:
- Accuracy
- Macro F1-score
- Confusion matrix

A persistence baseline was also evaluated for comparison.

## 5. Selected Model Results

### Hist Gradient Boosting - Flow Forecasting

| Horizon | MAE | RMSE | R2 |
|---|---:|---:|---:|
| 30 min | 20.98 | 31.11 | 0.9604 |
| 60 min | 35.17 | 50.47 | 0.8958 |
| 90 min | 48.79 | 68.31 | 0.8090 |
| 120 min | 61.03 | 83.33 | 0.7157 |

### Hist Gradient Boosting - Speed Forecasting

| Horizon | MAE | RMSE | R2 |
|---|---:|---:|---:|
| 30 min | 1.55 | 3.42 | 0.8189 |
| 60 min | 2.31 | 4.98 | 0.6157 |
| 90 min | 2.84 | 5.86 | 0.4672 |
| 120 min | 3.24 | 6.45 | 0.3549 |

### Congestion Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 0.9103 | 0.7702 |
| Hist Gradient Boosting | 0.9222 | 0.7228 |
| Linear SVM | 0.8920 | 0.5132 |

## 6. Model Artifacts

The trained Hist Gradient Boosting models are included in the public repository.

Random Forest models are generated locally using:

`src/train_pems04_random_forest_only.py`

Due to their large file sizes, Random Forest artifacts are excluded from Git.

## 7. Limitations

- The dataset is not mapped to the PRD's 25 named arterial road segments.
- Congestion classes are derived from speed thresholds.
- Classification macro F1 remains below the 0.80 target.
- Weather, accident and event data are unavailable.
- Live traffic forecasting and validated prediction uncertainty are not implemented.
- LSTM and GRU models are not implemented.
- Forecast performance generally decreases at longer horizons.

## 8. Reproducibility

Install dependencies:

`pip install -r requirements.txt`

Prepare the PEMS04 dataset using the project's data preparation script after placing the raw dataset at:

`data/raw/pems04/pems04.npz`

Run the dashboard:

`python -m streamlit run dashboard/app.py`

Detailed metrics are available in the `reports/` directory.
