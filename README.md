# 🏭 Predictive Maintenance — Machine Health Monitoring & RUL Prediction

An end-to-end **Predictive Maintenance Machine Learning project** for manufacturing and industrial applications.

This project uses machine sensor data to predict **Remaining Useful Life (RUL)** and identify potential machine degradation before failure occurs.

The project compares traditional Machine Learning models — **Random Forest and XGBoost** — with a deep learning **LSTM time-series model**, and deploys the final solution through an interactive **Streamlit dashboard**.

---

## 📌 Project Overview

### Business Problem

Unexpected machine failures can cause:

* Production downtime
* Expensive maintenance
* Equipment damage
* Reduced operational efficiency
* Safety risks

Traditional maintenance approaches are often:

**Reactive Maintenance**

> Repair the machine after it fails.

or

**Preventive Maintenance**

> Perform maintenance at predefined intervals.

This project implements:

### Predictive Maintenance

> Use historical sensor data and machine learning to estimate when a machine may fail.

The central prediction target is:

**Remaining Useful Life (RUL)**

For example:

```text
Current Cycle: 150
Estimated Failure Cycle: 190

Remaining Useful Life = 40 cycles
```

---

# 🎯 Project Objectives

The main objectives are:

* Analyze industrial machine sensor data
* Understand machine degradation patterns
* Engineer useful predictive features
* Predict Remaining Useful Life
* Compare multiple machine learning algorithms
* Build a time-series LSTM model
* Analyze prediction errors
* Develop an interactive monitoring dashboard
* Prepare the project for containerized deployment

---

# 📊 Dataset

## NASA C-MAPSS Turbofan Engine Dataset

The primary dataset used in this project is the **NASA C-MAPSS Turbofan Engine Degradation Dataset**.

The dataset simulates multiple aircraft engines operating until failure.

Each engine contains multiple operating cycles and sensor measurements.

### Dataset Components

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### Features

The dataset contains:

* Engine ID
* Cycle
* 3 operational settings
* 21 sensor measurements

Examples include:

```text
Temperature
Pressure
Speed
Vibration-related measurements
Operational parameters
```

---

# 🧠 Machine Learning Approach

The project follows a progressive modeling approach.

```text
Raw Sensor Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
RUL Calculation
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
Train / Validation Split
       ↓
Random Forest
       ↓
XGBoost
       ↓
LSTM
       ↓
Model Evaluation
       ↓
Streamlit Dashboard
```

---

# 🔬 Project Phases

## Phase 1 — Data Loading

Loaded the NASA turbofan engine datasets and inspected their structure.

Tasks included:

* Loading training data
* Loading test data
* Loading RUL ground truth
* Understanding dataset dimensions

---

## Phase 2 — Exploratory Data Analysis

Performed exploratory analysis to understand:

* Engine behavior
* Sensor distributions
* Operating conditions
* Sensor trends over time
* Correlation between sensors and RUL
* Engine degradation patterns

---

## Phase 3 — Feature Engineering

Calculated **Remaining Useful Life (RUL)** for the training data.

RUL was calculated based on the difference between the maximum operating cycle of an engine and its current cycle.

```text
RUL = Maximum Cycle - Current Cycle
```

Additional feature engineering included identifying informative sensors and preparing the data for machine learning models.

---

## Phase 3.5 — Feature Scaling

Applied feature scaling using `StandardScaler`.

The scaler was fitted only on the training data to avoid data leakage.

The trained scaler was later reused for validation, testing, and dashboard inference.

---

## Phase 4 — Baseline Machine Learning Models

Three baseline approaches were evaluated:

### Linear Regression

Used as a simple regression baseline.

### Random Forest

A tree-based ensemble model capable of capturing nonlinear relationships.

### XGBoost

A gradient boosting algorithm optimized for structured/tabular data.

### Validation Results

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     25.17 |     31.68 |     0.767 |
| Random Forest     |     23.96 |     31.62 |     0.768 |
| **XGBoost**       | **23.59** | **31.26** | **0.773** |

XGBoost performed best among the initial traditional ML models.

---

# 🚀 Phase 5–7 — Model Improvement

The project continued with:

* Feature importance analysis
* XGBoost hyperparameter tuning
* Validation performance comparison
* Error analysis
* Residual analysis

The objective was to improve RUL prediction while maintaining good generalization.

---

# 🧠 LSTM Model

Because machine sensor data is sequential, an LSTM model was introduced.

LSTM is particularly suitable for predictive maintenance because the current machine state depends not only on the current sensor readings but also on **historical sensor behavior**.

### Sequence Configuration

```text
Window Size = 30 cycles
Features    = 17
```

Input shape:

```text
(samples, 30, 17)
```

The model architecture included:

```text
LSTM
  ↓
Dropout
  ↓
Dense
  ↓
Output
```

### Model

```text
LSTM(64)
Dropout(0.2)
Dense(32, ReLU)
Dense(1)
```

Training included:

* Early stopping
* Learning-rate reduction
* Validation monitoring

---

# 📈 LSTM Validation Performance

The best validation experiment achieved approximately:

| Metric   |    Result |
| -------- | --------: |
| **MAE**  | **16.84** |
| **RMSE** | **22.96** |
| **R²**   | **0.845** |

This significantly improved upon the initial traditional ML validation results.

---

# ⚠️ Generalization Analysis

The final LSTM model was also evaluated on unseen test data.

### Final Test Performance

| Metric | Result |
| ------ | -----: |
| MAE    |  29.34 |
| RMSE   |  40.62 |
| R²     |  0.446 |

The difference between validation and test performance revealed a significant **generalization gap**.

This analysis was important because high validation performance alone does not guarantee strong performance on unseen operating conditions.

---

# 🔍 RUL Range Error Analysis

Prediction performance was analyzed across different RUL ranges.

| RUL Range |      MAE |
| --------- | -------: |
| 0–30      | **3.91** |
| 31–60     |    13.40 |
| 61–100    |    25.12 |
| 100+      |    33.79 |

### Key Observation

The model performs particularly well when the machine is close to failure.

For:

```text
RUL = 0–30
```

the model achieved approximately:

```text
MAE = 3.91 cycles
```

However, prediction errors increase substantially for machines with larger remaining lifetimes.

This suggests that the model is better at recognizing **late-stage degradation** than accurately estimating very long-term remaining life.

---

# 📊 Model Comparison

The project evaluates three major machine learning approaches:

| Model             | Type              | Strength                        |
| ----------------- | ----------------- | ------------------------------- |
| Linear Regression | Traditional ML    | Baseline                        |
| Random Forest     | Ensemble ML       | Nonlinear relationships         |
| XGBoost           | Gradient Boosting | Strong tabular performance      |
| LSTM              | Deep Learning     | Sequential/time-series patterns |

### Key Finding

Traditional tree-based models perform strongly on engineered tabular features, while LSTM provides an advantage in capturing temporal degradation patterns.

However, model evaluation on truly unseen data is critical because validation performance can overestimate real-world performance.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive **Streamlit-based predictive maintenance dashboard**.

The dashboard is designed to simulate an industrial machine monitoring system.

### Dashboard Features

#### 📌 Machine Health

Displays the current estimated machine health.

#### ⏳ Remaining Useful Life

Shows predicted RUL.

Example:

```text
RUL
42 cycles
```

#### ⚠️ Risk Level

Provides an interpretable machine condition:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

#### 🏭 Machine Overview

Displays overall machine status and operational information.

#### 📡 Sensor Health Cards

Displays individual sensor conditions and health indicators.

#### 📈 Prediction History

Visualizes RUL predictions over time.

---

# 🧩 Dashboard Architecture

```text
User Input
    ↓
Sensor Data
    ↓
Feature Preprocessing
    ↓
Saved StandardScaler
    ↓
30-Cycle Sequence
    ↓
Saved LSTM Model
    ↓
RUL Prediction
    ↓
Health / Risk Calculation
    ↓
Streamlit Dashboard
```

---

# 💾 Saved Model

The dashboard uses the trained model and preprocessing artifacts.

```text
models/
│
├── lstm_model.keras
└── scaler.pkl
```

The model expects:

```text
30 time steps
17 features
```

Input shape:

```text
(1, 30, 17)
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Deployment / Dashboard

* Streamlit
* Docker

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📁 Project Structure

```text
Predictive-Maintenance/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_lstm.ipynb
│   └── 07_model_evaluation.ipynb
│
├── models/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone [https://github.com/patelkaran952-hue/PredictiveMaintenance]
```

Navigate to the project:

```bash
cd PredictiveMaintenance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Start Streamlit:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

# 📌 Key Project Insights

### 1. Machine degradation is sequential

Sensor values should not always be treated as independent observations.

Historical sensor behavior contains valuable information about future machine health.

### 2. Feature engineering matters

Removing irrelevant or constant sensors and selecting meaningful features improves model efficiency.

### 3. Tree-based models are strong baselines

XGBoost provided strong performance on engineered tabular features.

### 4. LSTM captures temporal behavior

The LSTM model achieved better validation performance by learning patterns across sequences of machine cycles.

### 5. Validation performance isn't enough

The difference between validation and test results demonstrates the importance of evaluating models on truly unseen data.

### 6. Near-failure prediction is highly accurate

The LSTM performed particularly well for machines approaching failure, making it potentially useful for maintenance warning systems.

---

# 🎯 Business Impact

A production predictive maintenance system could help manufacturers:

* Reduce unplanned downtime
* Schedule maintenance proactively
* Reduce maintenance costs
* Improve equipment utilization
* Increase production reliability
* Improve machine safety
* Extend equipment lifetime

---

# 🔮 Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Better sequence construction
* Engine-level validation
* Attention-based LSTM
* GRU comparison
* Transformer-based time-series models
* Anomaly detection
* Failure classification
* Real-time IoT sensor integration
* Automated maintenance alerts
* Cloud deployment
* Model monitoring
* CI/CD pipeline
* Production API using FastAPI

---

# 📚 Project Status

```text
✅ Phase 1  — Data Loading
✅ Phase 2  — EDA
✅ Phase 3  — RUL Calculation
✅ Phase 3.5 — Feature Scaling
✅ Phase 4  — Baseline ML
✅ Phase 5  — XGBoost
✅ Phase 6  — Hyperparameter Tuning
✅ Phase 7  — LSTM
✅ Phase 8  — Model Evaluation
✅ Phase 9  — Test & Error Analysis
✅ Phase 10 — Streamlit Dashboard
🚧 Phase 12 — Deployment
```

---

# 👨‍💻 Author

**Karan Patel**

Aspiring Data Analyst / Machine Learning Engineer

Skills demonstrated in this project:

```text
Python
Pandas
NumPy
Scikit-learn
Random Forest
XGBoost
TensorFlow
LSTM
Time-Series ML
Feature Engineering
Streamlit
GitHub
```

---

# ⭐ If You Find This Project Useful

Give the repository a ⭐ on GitHub and feel free to explore the implementation.
