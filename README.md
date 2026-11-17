# FlowCast AI - Multi-Sensor Traffic Forecasting

## Project Overview

FlowCast AI is a machine learning project for short-term, multi-sensor traffic forecasting using the PEMS04 dataset.

The project aggregates five-minute sensor observations into 30-minute intervals and predicts traffic flow and average speed at four future horizons: 30, 60, 90 and 120 minutes.

It also provides a speed-based congestion classification and an interactive Streamlit dashboard.

## Key Features

- Multi-sensor traffic data processing using 307 PEMS04 sensors.
- 30-minute traffic data aggregation.
- Traffic flow forecasting at 30, 60, 90 and 120 minutes.
- Average speed forecasting at 30, 60, 90 and 120 minutes.
- Four-level speed-based congestion classification.
- Random Forest and Hist Gradient Boosting model experiments.
- Model evaluation using MAE, RMSE, R-squared and classification metrics.
- Interactive Streamlit dashboard with sensor selection and forecast visualizations.
- CSV download of forecast results.
- Feature importance visualization using permutation importance.

## Dataset

The primary dataset is PEMS04, containing traffic observations from 307 sensors at five-minute intervals.

The processed data included in this repository is:
- `pems04_30min_forecasting.csv.gz`
- `pems04_congestion_classification.csv.gz`

The raw PEMS04 dataset is not included. The processing script expects the original dataset at:

`data/raw/pems04/pems04.npz`

Obtain the dataset separately and place it at that path before running the data preparation script.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- XGBoost (for the earlier UCI traffic experiments)

## Model Performance

The following are the Hist Gradient Boosting results from the PEMS04 experiments.

### Traffic Flow Forecasting

| Horizon | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| 30 minutes | 20.98 | 31.11 | 0.9604 |
| 60 minutes | 35.17 | 50.47 | 0.8958 |
| 90 minutes | 48.79 | 68.31 | 0.8090 |
| 120 minutes | 61.03 | 83.33 | 0.7157 |

### Average Speed Forecasting

| Horizon | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| 30 minutes | 1.55 | 3.42 | 0.8189 |
| 60 minutes | 2.31 | 4.98 | 0.6157 |
| 90 minutes | 2.84 | 5.86 | 0.4672 |
| 120 minutes | 3.24 | 6.45 | 0.3549 |

### Congestion Classification

The Hist Gradient Boosting congestion classifier achieved:
- Accuracy: 0.9222
- Macro F1-score: 0.7228

Congestion labels are derived from speed thresholds:
- Free-flow: speed >= 60
- Moderate: speed >= 50 and < 60
- Heavy: speed >= 40 and < 50
- Severe: speed < 40

These are proxy labels, not independently verified congestion or incident records.

Detailed evaluation results are available in the `reports/` directory.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ganeshpisal6/flowcast-ai.git
cd flowcast-ai
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Dashboard

```powershell
streamlit run dashboard/app.py
```

The public repository includes the Hist Gradient Boosting model files. The larger Random Forest models are retained in the local development environment and are not included in this repository.

## Project Structure

```text
flowcast-ai/
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
│       ├── pems04_30min_forecasting.csv.gz
│       └── pems04_congestion_classification.csv.gz
├── models/
│   └── pems04_hist_gradient_boosting_*.joblib
├── reports/
│   ├── data_availability_and_limitations.md
│   ├── pems04_congestion_metrics.csv
│   ├── pems04_flow_all_horizons_metrics.csv
│   └── pems04_speed_all_horizons_metrics.csv
├── src/
│   ├── prepare_pems04_data.py
│   ├── train_pems04_models.py
│   ├── train_pems04_speed_models.py
│   └── train_pems04_congestion.py
├── requirements.txt
└── README.md
```

## Limitations

- The dashboard uses historical processed data, not live traffic feeds.
- The PEMS04 dataset has 307 sensors; it is not mapped here to 25 named arterial road segments.
- Verified calendar timestamps, weather observations, road capacity, accident records and event data are not included in the current pipeline.
- Congestion classes are generated from speed thresholds.
- Congestion classification macro F1-score is below the project's 0.80 target.
- Accident-risk prediction, travel-time estimation using verified road capacity, and live data integration are not implemented.
- Forecast accuracy can decrease at longer prediction horizons.

## Project Objective

To demonstrate multi-sensor traffic forecasting, machine learning model evaluation and interactive traffic analysis through a reproducible Python and Streamlit project.

## Author

Ganesh Pisal

## PRD Implementation Status

See [PRD Implementation Status](reports/PRD_IMPLEMENTATION_STATUS.md) for implementation coverage, model results and limitations.
