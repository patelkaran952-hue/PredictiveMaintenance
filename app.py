import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("lstm_rul_model.keras")
SCALER_PATH = Path("scaler_lstm.pkl")
DATA_PATH = Path("test_features.csv")

WINDOW_SIZE = 30

# IMPORTANT:
# These are the exact 17 features used by the LSTM.
# engine_id and cycle are NOT model features.

FEATURE_COLS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


IMPORTANT_SENSORS = [
    "sensor_11",
    "sensor_4",
    "sensor_12",
    "sensor_7",
    "sensor_15",
    "sensor_21",
]


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE CHECK
# ============================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append(str(MODEL_PATH))

if not SCALER_PATH.exists():
    missing_files.append(str(SCALER_PATH))

if not DATA_PATH.exists():
    missing_files.append(str(DATA_PATH))


if missing_files:
    st.error("❌ Required files are missing.")

    st.write("The following files could not be found:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Make sure app.py, lstm_model.keras, scaler.pkl "
        "and test.csv are in the same folder."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ============================================================
# LOAD TEST DATA
# ============================================================


@st.cache_data
def load_test_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    model = load_lstm_model()

    scaler = load_scaler()

    test_df = load_test_data()

except Exception as e:
    st.error("❌ Error loading model, scaler or dataset.")

    st.exception(e)

    st.stop()


# ============================================================
# VALIDATE DATA COLUMNS
# ============================================================

required_columns = ["engine_id", "cycle"] + FEATURE_COLS


missing_columns = [col for col in required_columns if col not in test_df.columns]


if missing_columns:
    st.error("❌ Required columns are missing from test.csv.")

    st.write(missing_columns)

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

test_df = test_df.copy()

test_df = test_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


# ============================================================
# MODEL VALIDATION
# ============================================================

try:
    model_timesteps = model.input_shape[-2]

    model_features = model.input_shape[-1]

except Exception as e:
    st.error("❌ Could not determine model input shape.")

    st.exception(e)

    st.stop()


# Sequence length

if model_timesteps != WINDOW_SIZE:
    st.error(
        f"""
❌ Sequence length mismatch.

Model expects: {model_timesteps}
Dashboard uses: {WINDOW_SIZE}
"""
    )

    st.stop()


# Number of features

if model_features != len(FEATURE_COLS):
    st.error(
        f"""
❌ Feature mismatch.

Model expects: {model_features}
Dashboard uses: {len(FEATURE_COLS)}
"""
    )

    st.stop()


# Scaler

if scaler.n_features_in_ != len(FEATURE_COLS):
    st.error(
        f"""
❌ Scaler feature mismatch.

Scaler expects: {scaler.n_features_in_}
Dashboard uses: {len(FEATURE_COLS)}
"""
    )

    st.stop()


# ============================================================
# RUL PREDICTION FUNCTION
# ============================================================


def predict_rul(engine_df):
    engine_df = engine_df.sort_values("cycle").copy()

    if len(engine_df) < WINDOW_SIZE:
        return None

    # Last 30 cycles
    latest_window = engine_df[FEATURE_COLS].tail(WINDOW_SIZE)

    # Convert to numpy
    X = latest_window.values

    # Scale
    X_scaled = scaler.transform(X)

    # Reshape for LSTM
    X_lstm = X_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))

    # Prediction
    prediction = model.predict(X_lstm, verbose=0)

    predicted_rul = float(prediction[0][0])

    # RUL cannot be negative
    predicted_rul = max(predicted_rul, 0)

    return predicted_rul


# ============================================================
# PREDICTION HISTORY
# ============================================================


def generate_prediction_history(engine_df):
    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

    if len(engine_df) < WINDOW_SIZE:
        return pd.DataFrame(columns=["Cycle", "Predicted RUL"])

    sequences = []

    cycles = []

    # --------------------------------------------------------
    # CREATE 30-CYCLE WINDOWS
    # --------------------------------------------------------

    for end_idx in range(WINDOW_SIZE, len(engine_df) + 1):
        window = engine_df.iloc[end_idx - WINDOW_SIZE : end_idx]

        X_window = window[FEATURE_COLS].values

        sequences.append(X_window)

        cycles.append(int(window["cycle"].iloc[-1]))

    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------

    X_sequences = np.array(sequences, dtype=np.float32)

    n_sequences = X_sequences.shape[0]

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    X_flat = X_sequences.reshape(-1, len(FEATURE_COLS))

    X_flat_scaled = scaler.transform(X_flat)

    # --------------------------------------------------------
    # RESTORE LSTM SHAPE
    # --------------------------------------------------------

    X_scaled = X_flat_scaled.reshape(n_sequences, WINDOW_SIZE, len(FEATURE_COLS))

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(X_scaled, batch_size=64, verbose=0).flatten()

    predictions = np.maximum(predictions, 0)

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    history_df = pd.DataFrame({"Cycle": cycles, "Predicted RUL": predictions})

    return history_df


# ============================================================
# RISK CALCULATION
# ============================================================


def calculate_risk(predicted_rul):
    if predicted_rul <= 30:
        risk = "CRITICAL"

        icon = "🔴"

        health = 20

        message = "Immediate maintenance recommended."

    elif predicted_rul <= 60:
        risk = "HIGH"

        icon = "🟠"

        health = 40

        message = "Maintenance should be planned soon."

    elif predicted_rul <= 100:
        risk = "MEDIUM"

        icon = "🟡"

        health = 65

        message = "Continue monitoring the machine."

    else:
        risk = "LOW"

        icon = "🟢"

        health = 85

        message = "Machine operating normally."

    return (risk, icon, health, message)


# ============================================================
# SENSOR HEALTH
# ============================================================


def get_sensor_status(engine_df, sensor_name):
    values = pd.to_numeric(engine_df[sensor_name], errors="coerce").dropna()

    if len(values) == 0:
        return (0, "Critical", "✖", 0)

    latest_value = float(values.iloc[-1])

    mean_value = float(values.mean())

    std_value = float(values.std())

    # Avoid division by zero
    if std_value == 0 or np.isnan(std_value):
        z_score = 0

    else:
        z_score = abs((latest_value - mean_value) / std_value)

    if z_score >= 2.5:
        status = "Critical"

        icon = "✖"

    elif z_score >= 1.5:
        status = "Warning"

        icon = "⚠"

    else:
        status = "Normal"

        icon = "✓"

    return (latest_value, status, icon, z_score)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("---")


# ============================================================
# MACHINE SELECTION
# ============================================================

engine_ids = sorted(test_df["engine_id"].unique())


machine_id = st.sidebar.selectbox(
    "Select Machine", engine_ids, format_func=lambda x: f"Engine {x}"
)


st.sidebar.markdown("---")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.sidebar.subheader("Model Information")

st.sidebar.write("Model: LSTM")

st.sidebar.write(f"Sequence Length: {WINDOW_SIZE}")

st.sidebar.write(f"Features: {len(FEATURE_COLS)}")

st.sidebar.write("Scaler: StandardScaler")

st.sidebar.markdown("---")

st.sidebar.success("Model Status: Ready")


# ============================================================
# SELECTED ENGINE DATA
# ============================================================

selected_engine = test_df[test_df["engine_id"] == machine_id].copy()


selected_engine = selected_engine.sort_values("cycle").reset_index(drop=True)


# ============================================================
# CHECK ENGINE DATA
# ============================================================

if len(selected_engine) < WINDOW_SIZE:
    st.error(
        f"""
❌ Engine {machine_id} does not have enough cycles.

Available cycles: {len(selected_engine)}

Required cycles: {WINDOW_SIZE}
"""
    )

    st.stop()


# ============================================================
# CURRENT CYCLE
# ============================================================

current_cycle = int(selected_engine["cycle"].max())


# ============================================================
# CURRENT RUL
# ============================================================

predicted_rul = predict_rul(selected_engine)


if predicted_rul is None:
    st.error("❌ Unable to generate RUL prediction.")

    st.stop()


# ============================================================
# RISK / HEALTH
# ============================================================

(risk_level, machine_status, health_score, maintenance_message) = calculate_risk(
    predicted_rul
)


# ============================================================
# HEADER
# ============================================================

st.title("🏭 Predictive Maintenance Dashboard")

st.caption("Machine Health Monitoring & Remaining Useful Life Prediction")


# ============================================================
# MACHINE OVERVIEW
# ============================================================

st.subheader("Machine Overview")

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# RUL
# ------------------------------------------------------------

with col1:
    with st.container(border=True):
        st.write("Remaining Useful Life")

        st.metric(label="", value=f"{predicted_rul:.1f}")

        st.caption("cycles remaining")


# ------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------

with col2:
    with st.container(border=True):
        st.write("Machine Health")

        st.metric(label="", value=f"{health_score}%")

        st.caption("estimated health")


# ------------------------------------------------------------
# RISK
# ------------------------------------------------------------

with col3:
    with st.container(border=True):
        st.write("Risk Level")

        st.metric(label="", value=risk_level)

        st.caption("maintenance risk")


# ------------------------------------------------------------
# MACHINE STATUS
# ------------------------------------------------------------

with col4:
    with st.container(border=True):
        st.write("Machine Status")

        st.metric(label="", value=machine_status)

        st.caption(f"Engine {machine_id} • Cycle {current_cycle}")


# ============================================================
# REMAINING USEFUL LIFE
# ============================================================

st.subheader("Remaining Useful Life")

rul_col, prediction_col = st.columns([2, 1])


# ============================================================
# RUL DISPLAY
# ============================================================

with rul_col:
    with st.container(border=True):
        st.markdown("### Predicted Remaining Useful Life")

        st.metric(label="", value=f"{predicted_rul:.1f} cycles")

        # ----------------------------------------------------
        # RUL PROGRESS
        # ----------------------------------------------------

        # Dashboard visualization scale:
        # 200 cycles = 100%

        progress_value = min(predicted_rul / 200, 1.0)

        st.progress(progress_value)

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if risk_level == "CRITICAL":
            st.error("🔴 CRITICAL RISK")

        elif risk_level == "HIGH":
            st.warning("🟠 HIGH RISK")

        elif risk_level == "MEDIUM":
            st.warning("🟡 MEDIUM RISK")

        else:
            st.success("🟢 LOW RISK")

        st.caption(maintenance_message)


# ============================================================
# PREDICTION STATUS
# ============================================================

with prediction_col:
    with st.container(border=True):
        st.markdown("### Prediction Status")

        st.success("● Active")

        st.write(f"**Model:** LSTM")

        st.write(f"**Machine:** Engine {machine_id}")

        st.write(f"**Current Cycle:** {current_cycle}")

        st.write(f"**Input Window:** {WINDOW_SIZE} cycles")

        st.write(f"**Features:** {len(FEATURE_COLS)}")

        st.write("**Prediction:** RUL")


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

st.subheader("Maintenance Recommendation")


if risk_level == "CRITICAL":
    st.error(
        f"🔴 **{maintenance_message}** Predicted RUL: **{predicted_rul:.1f} cycles**"
    )

elif risk_level == "HIGH":
    st.warning(
        f"🟠 **{maintenance_message}** Predicted RUL: **{predicted_rul:.1f} cycles**"
    )

elif risk_level == "MEDIUM":
    st.warning(
        f"🟡 **{maintenance_message}** Predicted RUL: **{predicted_rul:.1f} cycles**"
    )

else:
    st.success(
        f"🟢 **{maintenance_message}** Predicted RUL: **{predicted_rul:.1f} cycles**"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.subheader("Prediction History")


with st.spinner("Generating prediction history..."):
    history_df = generate_prediction_history(selected_engine)


if not history_df.empty:
    st.line_chart(history_df, x="Cycle", y="Predicted RUL", use_container_width=True)

else:
    st.info("Not enough cycles to generate prediction history.")


# ============================================================
# SENSOR HEALTH
# ============================================================

st.subheader("Sensor Health")


sensor_cols = st.columns(3)


for i, sensor_name in enumerate(IMPORTANT_SENSORS):
    (sensor_value, sensor_status, sensor_icon, z_score) = get_sensor_status(
        selected_engine, sensor_name
    )

    with sensor_cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"### {sensor_name.replace('_', ' ').title()}")

            st.metric(label="Current Value", value=f"{sensor_value:.3f}")

            if sensor_status == "Normal":
                st.success(f"✓ {sensor_status}")

            elif sensor_status == "Warning":
                st.warning(f"⚠ {sensor_status}")

            else:
                st.error(f"✖ {sensor_status}")


# ============================================================
# SENSOR TRENDS
# ============================================================

st.subheader("Sensor Trends")


trend_sensor = st.selectbox(
    "Select Sensor",
    IMPORTANT_SENSORS,
    format_func=lambda x: x.replace("_", " ").title(),
)


sensor_trend_df = selected_engine[["cycle", trend_sensor]].copy()


st.line_chart(sensor_trend_df, x="cycle", y=trend_sensor, use_container_width=True)


# ============================================================
# MACHINE INFORMATION
# ============================================================

st.subheader("Machine Information")


info1, info2, info3, info4 = st.columns(4)


with info1:
    st.metric("Machine", f"Engine {machine_id}")


with info2:
    st.metric("Current Cycle", current_cycle)


with info3:
    st.metric("Available Cycles", len(selected_engine))


with info4:
    st.metric("Predicted RUL", f"{predicted_rul:.1f}")


# ============================================================
# LATEST MACHINE DATA
# ============================================================

with st.expander("🔍 View Latest Machine Data"):
    display_cols = ["engine_id", "cycle"] + FEATURE_COLS

    st.dataframe(selected_engine[display_cols].tail(10), use_container_width=True)


# ============================================================
# MODEL VALIDATION
# ============================================================

with st.expander("🧪 Model Validation"):
    st.write("Model input shape:", model.input_shape)

    st.write("Scaler features:", scaler.n_features_in_)

    st.write("Dashboard sequence:", f"(1, {WINDOW_SIZE}, {len(FEATURE_COLS)})")

    st.write("Selected engine:", machine_id)

    st.write("Current cycle:", current_cycle)

    st.success("✅ LSTM dashboard validation passed.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Predictive Maintenance System • LSTM-based Remaining Useful Life Prediction"
)
