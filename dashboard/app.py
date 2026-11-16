from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "pems04_30min_forecasting.csv.gz"
CONGESTION_FILE = ROOT / "data" / "processed" / "pems04_congestion_classification.csv.gz"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = ["flow", "occupancy", "speed"]
CONGESTION_FEATURES = ["flow", "occupancy", "sensor_id", "interval_index"]
HORIZONS = [30, 60, 90, 120]

MODEL_CANDIDATES = ["random_forest", "hist_gradient_boosting"]
MODEL_OPTIONS = [
    model for model in MODEL_CANDIDATES
    if (MODEL_DIR / f"pems04_{model}_congestion.joblib").exists()
    and all(
        (MODEL_DIR / f"pems04_{model}_{target}_{horizon}min.joblib").exists()
        for target in ["flow", "speed"]
        for horizon in HORIZONS
    )
]

st.set_page_config(page_title="FlowCast AI | PEMS04", page_icon="F", layout="wide")
st.title("FlowCast AI")
st.subheader("Multi-Sensor Traffic Forecasting Dashboard")
st.caption("PEMS04 | 307 sensors | 30-minute intervals | Flow, speed and congestion forecasting")

if not MODEL_OPTIONS:
    st.error("No complete PEMS04 model set found. Please train or add the required models.")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df.sort_values(["sensor_id", "interval_index"]).reset_index(drop=True)


@st.cache_data
def load_congestion_data():
    return pd.read_csv(CONGESTION_FILE)


@st.cache_data
def load_metrics(filename):
    path = REPORT_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


if not DATA_FILE.exists():
    st.error(f"PEMS04 forecasting dataset not found: {DATA_FILE}")
    st.stop()
if not CONGESTION_FILE.exists():
    st.error(f"Congestion dataset not found: {CONGESTION_FILE}")
    st.stop()

try:
    df = load_data()
    congestion_df = load_congestion_data()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()

required = FEATURES + ["sensor_id", "interval_index"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Required dataset columns are missing: {missing}")
    st.stop()

st.sidebar.header("Dashboard Settings")
sensors = sorted(df["sensor_id"].dropna().unique().tolist())
selected_sensor = st.sidebar.selectbox(
    "Select traffic sensor", sensors, format_func=lambda x: f"Sensor {int(x)}"
)
selected_model = st.sidebar.selectbox(
    "Select forecasting model", MODEL_OPTIONS,
    format_func=lambda x: x.replace("_", " ").title()
)

if st.sidebar.button("Refresh dashboard"):
    load_data.clear()
    load_congestion_data.clear()
    load_metrics.clear()
    load_model.clear()
    st.rerun()

sensor_data = df[df["sensor_id"] == selected_sensor].sort_values("interval_index")
if sensor_data.empty:
    st.error("No records found for the selected sensor.")
    st.stop()

latest = sensor_data.iloc[-1]
input_row = latest[FEATURES].to_frame().T.astype(float)
st.sidebar.markdown("---")
st.sidebar.write(f"**Selected sensor:** {int(selected_sensor)}")
st.sidebar.write(f"**Latest interval index:** {int(latest['interval_index'])}")
st.sidebar.caption("One interval represents 30 minutes.")

st.markdown("## 1. Current Sensor Readings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sensor ID", int(selected_sensor))
c2.metric("Current flow", f"{latest['flow']:.2f}")
c3.metric("Current speed", f"{latest['speed']:.2f}")
c4.metric("Occupancy", f"{latest['occupancy']:.4f}")
st.caption("Latest available historical observations; these are not live roadside measurements.")

st.divider()
st.markdown("## 2. Flow and Speed Forecasts")
forecast_rows = []
prediction_error = None

for horizon in HORIZONS:
    for target_type in ["flow", "speed"]:
        filename = f"pems04_{selected_model}_{target_type}_{horizon}min.joblib"
        try:
            model = load_model(filename)
            prediction = float(model.predict(input_row[FEATURES])[0])
            forecast_rows.append({
                "Horizon": f"{horizon} minutes",
                "Horizon (minutes)": horizon,
                "Target": target_type.title(),
                "Prediction": max(0.0, prediction),
                "Model": selected_model.replace("_", " ").title(),
            })
        except Exception as exc:
            prediction_error = f"{filename}: {exc}"
            break
    if prediction_error:
        break

if prediction_error:
    st.error(f"Prediction could not be generated: {prediction_error}")
    st.stop()

forecast_df = pd.DataFrame(forecast_rows)
flow_forecasts = forecast_df[forecast_df["Target"] == "Flow"]
speed_forecasts = forecast_df[forecast_df["Target"] == "Speed"]

st.markdown("### Predicted Traffic Flow")

flow_table = flow_forecasts[["Horizon", "Prediction"]].rename(
    columns={"Prediction": "Predicted Flow"}
)
st.dataframe(flow_table, hide_index=True, use_container_width=True)

st.markdown("### Predicted Average Speed")

speed_table = speed_forecasts[["Horizon", "Prediction"]].rename(
    columns={"Prediction": "Predicted Speed"}
)
st.dataframe(speed_table, hide_index=True, use_container_width=True)
st.markdown("### Predicted average speed")
speed_cols = st.columns(4)
for col, (_, item) in zip(speed_cols, speed_forecasts.iterrows()):
    col.metric(item["Horizon"], f"{item['Prediction']:.2f}")

forecast_fig = px.line(
    forecast_df, x="Horizon (minutes)", y="Prediction", color="Target",
    markers=True, title="Flow and Speed Forecasts Across Four Horizons"
)
st.plotly_chart(forecast_fig, use_container_width=True)
st.download_button(
    "Download FlowCast Forecasts (CSV)",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name=f"flowcast_sensor_{int(selected_sensor)}_forecasts.csv",
    mime="text/csv",
)

st.divider()
st.markdown("## 3. Congestion Classification")
congestion_row = latest[CONGESTION_FEATURES].to_frame().T
congestion_model_name = f"pems04_{selected_model}_congestion.joblib"
try:
    congestion_model = load_model(congestion_model_name)
    predicted_class = congestion_model.predict(congestion_row)[0]
    st.metric("Predicted congestion level", str(predicted_class))
    if hasattr(congestion_model, "predict_proba"):
        probabilities = congestion_model.predict_proba(congestion_row)[0]
        probability_df = pd.DataFrame({
            "Congestion class": congestion_model.classes_,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False)
        probability_fig = px.bar(
            probability_df, x="Congestion class", y="Probability",
            title="Congestion Class Probabilities", text_auto=".3f"
        )
        st.plotly_chart(probability_fig, use_container_width=True)
except Exception as exc:
    st.error(f"Congestion prediction failed: {exc}")

st.caption(
    "Congestion labels are provisional speed-based proxy labels. "
    "Probabilities are model outputs, not independently calibrated confidence scores."
)

st.divider()
st.markdown("## 4. Historical Traffic Trends")
history = sensor_data.tail(96).copy()
history_fig = px.line(
    history, x="interval_index", y=["flow", "speed"],
    title="Recent Flow and Speed History - Selected Sensor"
)
st.plotly_chart(history_fig, use_container_width=True)

st.divider()
st.markdown("## 5. Multi-Sensor Congestion Heatmap")
latest_interval = int(df["interval_index"].max())
heat_data = congestion_df[
    congestion_df["interval_index"] >= latest_interval - 47
].copy()
if not heat_data.empty:
    heatmap = heat_data.pivot_table(
        index="sensor_id", columns="interval_index", values="speed", aggfunc="mean"
    ).sort_index()
    heat_fig = px.imshow(
        heatmap, aspect="auto",
        labels={"x": "30-minute interval index", "y": "Sensor ID", "color": "Speed"},
        title="Speed Heatmap Across Sensors", color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(heat_fig, use_container_width=True)
else:
    st.info("No records available for the heatmap.")

st.divider()
st.markdown("## 6. Sensor Comparison")
comparison = (
    df[df["interval_index"] == latest_interval][
        ["sensor_id", "flow", "speed", "occupancy"]
    ].sort_values("flow", ascending=False).head(20)
)
if not comparison.empty:
    comparison_fig = px.bar(
        comparison, x="sensor_id", y="flow",
        title="Top 20 Sensors by Flow at Latest Available Interval",
        labels={"sensor_id": "Sensor ID", "flow": "Traffic flow"}
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

st.divider()
st.markdown("## 7. Model Performance")
for heading, filename in [
    ("Flow forecasting metrics", "pems04_flow_all_horizons_metrics.csv"),
    ("Speed forecasting metrics", "pems04_speed_all_horizons_metrics.csv"),
    ("Congestion classification metrics", "pems04_congestion_metrics.csv"),
]:
    st.markdown(f"### {heading}")
    report = load_metrics(filename)
    if not report.empty:
        st.dataframe(report, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Evaluation report not found: {filename}")

st.divider()
st.markdown("## 8. Feature Importance")

try:
    importance_model = load_model(
        f"pems04_{selected_model}_flow_30min.joblib"
    )

    importance_data = df[FEATURES + ["flow_next_30min"]].dropna()

    importance_sample = importance_data.sample(
        n=min(2000, len(importance_data)),
        random_state=42
    )

    importance_result = permutation_importance(
        importance_model,
        importance_sample[FEATURES],
        importance_sample["flow_next_30min"],
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance_result.importances_mean,
        "Variation": importance_result.importances_std,
    }).sort_values("Importance", ascending=True)

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        error_x="Variation",
        orientation="h",
        title=f"{selected_model.replace('_', ' ').title()} - "
              "Permutation Feature Importance"
    )

    st.plotly_chart(importance_fig, use_container_width=True)
    st.caption(
        "Importance is measured by the increase in prediction error "
        "when each feature is shuffled. Higher values indicate greater "
        "importance. Calculated on a reproducible sample of up to 2,000 "
        "historical records."
    )

except Exception as exc:
    st.warning(f"Could not calculate feature importance: {exc}")

st.divider()
st.markdown("## 9. Data and Model Limitations")
st.info(
    "This dashboard uses historical PEMS04 data. It does not receive live traffic "
    "or weather feeds. PEMS04 has 307 sensors, not the PRD's 25 mapped road segments. "
    "Road distances, capacity, linked weather, calendar events and accident labels "
    "are not available in the inspected data. Therefore, travel-time estimation, "
    "weather analysis and accident-risk prediction are not presented as completed features."
)
st.caption(
    "FlowCast AI | PEMS04 multi-sensor forecasting | "
    "Forecast horizons: 30, 60, 90 and 120 minutes"
)
