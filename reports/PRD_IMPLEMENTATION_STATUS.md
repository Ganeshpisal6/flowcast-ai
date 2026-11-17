# FlowCast AI - PRD Implementation Status

## Completed
- PEMS04 processing using 307 sensors and 30-minute aggregation.
- Flow and speed forecasting at 30, 60, 90 and 120 minutes.
- Speed-based four-class congestion classification.
- Random Forest and Hist Gradient Boosting experiments.
- Linear Regression, Decision Tree and XGBoost comparisons for 30-minute targets.
- NumPy gradient-descent Linear Regression.
- Linear SVM congestion experiment.
- Persistence baseline comparison.
- Regression and classification evaluation reports.
- Streamlit dashboard with sensor selection, forecasts, visualizations and CSV export.

## Key results
Hist Gradient Boosting flow forecasting:
| Horizon | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| 30 min | 20.98 | 31.11 | 0.9604 |
| 60 min | 35.17 | 50.47 | 0.8958 |
| 90 min | 48.79 | 68.31 | 0.8090 |
| 120 min | 61.03 | 83.33 | 0.7157 |

Hist Gradient Boosting speed forecasting:
| Horizon | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| 30 min | 1.55 | 3.42 | 0.8189 |
| 60 min | 2.31 | 4.98 | 0.6157 |
| 90 min | 2.84 | 5.86 | 0.4672 |
| 120 min | 3.24 | 6.45 | 0.3549 |

Congestion classification:
| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Random Forest | 0.9103 | 0.7702 |
| Hist Gradient Boosting | 0.9222 | 0.7228 |
| Linear SVM | 0.8920 | 0.5132 |

NumPy Linear Regression for 30-minute flow:
- MAE: 21.2221
- RMSE: 31.5056
- R-squared: 0.9594
- 160,000 training samples and 500 epochs.

## Partial and unavailable requirements
- The dataset contains 307 sensors, not the PRD's 25 named arterial road segments.
- Weather and calendar/event integration are unavailable because linked observations are not provided.
- Travel-time estimation is not implemented because verified road distance and capacity are unavailable.
- Accident-risk prediction is not implemented because accident labels are unavailable.
- Live traffic ingestion is not implemented.
- LSTM/GRU from-scratch models are not implemented.
- Validated prediction confidence is not implemented.
- Upload, quarantine and retraining workflows are not fully implemented.
- Congestion macro-F1 target of 0.80 was not achieved.
- Accident ROC-AUC cannot be evaluated without accident labels.

## Data limitations
PEMS04 provides flow, occupancy and speed observations. Congestion classes are derived from speed thresholds and are proxy labels, not independently verified ground truth. Raw data does not provide the verified timestamps, weather, event, accident, road-capacity and road-mapping information needed for several original PRD modules.

## Reproducibility
Install dependencies with `pip install -r requirements.txt`.
Run the dashboard with `streamlit run dashboard/app.py`.
The raw PEMS04 dataset must be obtained separately and placed at `data/raw/pems04/pems04.npz` to reproduce data preparation.

## Overall assessment
The project implements the core historical multi-sensor forecasting workflow, model experiments, evaluation reports and interactive dashboard. The full original PRD is not completely satisfied because several requirements depend on unavailable datasets or additional models and workflows. These limitations are explicitly documented.
