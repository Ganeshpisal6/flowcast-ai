from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traffic_forecasting_v2.csv"
MODEL_DIR = ROOT / "models"
REPORT_FILE = ROOT / "reports" / "forecast_horizon_comparison.csv"

FEATURES = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "is_weekend",
    "traffic_lag_1h", "traffic_lag_2h", "traffic_lag_3h",
    "traffic_lag_4h", "traffic_lag_24h",
    "traffic_mean_previous_3h", "traffic_mean_previous_24h",
]

st.set_page_config(
    page_title="FlowCast AI | Traffic Forecasting",
    page_icon="F",
    layout="wide",
)

st.title("FlowCast AI")
st.subheader("Smart Traffic Flow Prediction System")
st.caption(
    "Machine-learning traffic forecasting using historical traffic "
    "patterns and weather features."
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    return df


@st.cache_resource
def load_forecast_model(horizon):
    path = MODEL_DIR / f"xgboost_traffic_{horizon}h.joblib"
    return joblib.load(path)


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the forecasting dataset: {exc}")
    st.stop()

missing = [feature for feature in FEATURES if feature not in df.columns]
if missing:
    st.error(f"Dataset is missing required features: {missing}")
    st.stop()

numeric_df = df.copy()
numeric_df[FEATURES] = numeric_df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)
valid_df = numeric_df.dropna(subset=FEATURES)

if valid_df.empty:
    st.error("No rows contain all required numeric forecasting features.")
    st.stop()

latest = valid_df.iloc[-1]
input_row = latest[FEATURES].to_frame().T.astype("float64")

st.sidebar.header("Forecast Settings")
st.sidebar.info(
    "This dashboard uses the latest valid row in the historical dataset. "
    "It does not fetch live traffic or weather data."
)
st.sidebar.write("**Latest input timestamp**")
st.sidebar.write(str(latest["date_time"]))

if st.sidebar.button("Refresh dashboard data"):
    load_data.clear()
    load_forecast_model.clear()
    st.rerun()

st.markdown("## Traffic Forecasts")

predictions = []
forecast_error = None

for horizon in [1, 2, 3, 4]:
    model_path = MODEL_DIR / f"xgboost_traffic_{horizon}h.joblib"

    if not model_path.exists():
        forecast_error = f"Missing model file: {model_path.name}"
        break

    try:
        saved = load_forecast_model(horizon)
        model = saved["model"]
        model_features = saved["features"]

        absent = [f for f in model_features if f not in input_row.columns]
        if absent:
            forecast_error = (
                f"The {horizon}-hour model requires unavailable features: "
                f"{absent}"
            )
            break

        value = float(model.predict(input_row[model_features])[0])
        forecast_time = latest["date_time"] + pd.Timedelta(hours=horizon)

        predictions.append({
            "Horizon": f"+{horizon} hour",
            "Hours ahead": horizon,
            "Forecast time": forecast_time,
            "Predicted traffic": max(0, value),
        })
    except Exception as exc:
        forecast_error = f"Could not generate the {horizon}-hour forecast: {exc}"
        break

if forecast_error:
    st.error(forecast_error)
    st.stop()

forecast_df = pd.DataFrame(predictions)

st.caption(
    f"Forecasts calculated from the historical input at "
    f"{latest['date_time']}."
)

cols = st.columns(4)
for col, item in zip(cols, predictions):
    with col:
        st.metric(
            label=item["Horizon"],
            value=f"{item['Predicted traffic']:,.0f}",
        )
        st.caption(item["Forecast time"].strftime("%d %b %Y, %H:%M"))

st.markdown("### Forecast comparison")

fig = px.bar(
    forecast_df,
    x="Horizon",
    y="Predicted traffic",
    text_auto=".0f",
    title="Predicted Traffic by Forecast Horizon",
    labels={
        "Horizon": "Forecast horizon",
        "Predicted traffic": "Predicted traffic volume",
    },
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download forecast results (CSV)",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="flowcast_forecast_results.csv",
    mime="text/csv",
)

st.divider()
st.markdown("## Historical Traffic Analysis")

traffic_column = next(
    (
        col for col in [
            "traffic_volume", "traffic", "traffic_count",
            "vehicles", "vehicle_count"
        ]
        if col in df.columns
    ),
    None,
)

if traffic_column:
    history = df[["date_time", traffic_column]].copy()
    history[traffic_column] = pd.to_numeric(
        history[traffic_column], errors="coerce"
    )
    history = history.dropna().tail(500)

    if not history.empty:
        history_fig = px.line(
            history,
            x="date_time",
            y=traffic_column,
            title="Recent Historical Traffic (Last 500 Available Records)",
            labels={
                "date_time": "Date and time",
                traffic_column: "Traffic volume",
            },
        )
        st.plotly_chart(history_fig, use_container_width=True)
    else:
        st.info("No numeric historical traffic values are available to plot.")
else:
    st.info(
        "The historical traffic column was not identified automatically. "
        "Check the dataset column names to enable this chart."
    )

st.divider()
st.markdown("## Model Evaluation")

if REPORT_FILE.exists():
    try:
        report = pd.read_csv(REPORT_FILE)
        st.dataframe(report, use_container_width=True, hide_index=True)

        st.caption(
            "These metrics come from the saved evaluation report. "
            "They describe the recorded test evaluation, not live performance."
        )
    except Exception as exc:
        st.warning(f"Could not read the model evaluation report: {exc}")
else:
    st.info(
        "No model comparison report found yet. Run "
        "python src\\train_all_horizons.py to generate it."
    )

st.divider()
st.markdown("## Project Information")

col1, col2, col3 = st.columns(3)
col1.metric("Historical records", f"{len(df):,}")
col2.metric("Forecast horizons", "4")
col3.metric("Input features", str(len(FEATURES)))

st.caption(
    "FlowCast AI | Traffic forecasting prototype | "
    "Predictions are based on historical dataset features."
)
