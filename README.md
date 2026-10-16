# FlowCast AI - Traffic Forecasting

## Project Overview
FlowCast AI is a machine learning project that predicts traffic volume for the next 1, 2, 3, and 4 hours using historical traffic data, weather information, and time-based features.

## Key Features
- Forecast traffic volume for four future time horizons.
- Train XGBoost regression models.
- Compare model performance using evaluation metrics.
- Visualize forecasts through an interactive Streamlit dashboard.
- View historical traffic trends.
- Download forecast results as a CSV file.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Matplotlib and Seaborn

## Project Structure
```text
flowcast-ai/
+-- dashboard/
¦   +-- app.py
+-- data/
¦   +-- raw/
¦   +-- processed/
+-- models/
+-- reports/
+-- src/
+-- requirements.txt
+-- README.md
```

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

## Train the Models

```powershell
python src/train_all_horizons.py
```

## Generate Predictions

```powershell
python src/predict_traffic.py
```

## Model Evaluation
The project evaluates forecasting performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). Evaluation reports are stored in the `reports/` directory.

## Dataset and Limitations
This project uses historical Metro Interstate Traffic Volume data. The dataset ends on September 30, 2018. Predictions are based on historical data and should not be interpreted as live traffic forecasts.

## Project Objective
To demonstrate the use of machine learning for short-term traffic volume forecasting and interactive traffic data analysis.

## Author
Ganesh Pisal
