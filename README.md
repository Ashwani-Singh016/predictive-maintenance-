# Predictive Maintenance for Industrial Machinery

A machine learning project that predicts equipment failure from real-time
sensor readings, enabling maintenance to be scheduled *before* a breakdown
happens rather than after.

## 1. Problem Statement

Unplanned equipment downtime is one of the costliest problems in
manufacturing. Traditional maintenance is either:
- **Reactive** — fix it after it breaks (expensive downtime, safety risk), or
- **Preventive** — fix it on a fixed schedule (wastes good parts, still misses failures)

**Predictive maintenance** uses sensor data and machine learning to estimate
the *probability of failure* in real time, so maintenance can be scheduled
exactly when it's needed — reducing both downtime and unnecessary servicing.

This project builds a binary classifier that predicts `Machine_failure`
(1 = will fail / needs attention, 0 = healthy) from five sensor readings,
and identifies *which* failure mechanism is most likely.

## 2. Dataset

The dataset schema and failure logic are modeled on the widely-used **AI4I
2020 Predictive Maintenance Dataset**, a standard benchmark in PdM
literature. Since no internet access was available to pull the original
file, `src/generate_data.py` procedurally generates a synthetic dataset of
**10,000 machine readings** using the same feature set and the same
physically-grounded failure conditions, so the project behaves like a
real industrial dataset while remaining fully reproducible (fixed random
seed).

| Column | Description |
|---|---|
| `Type` | Product quality variant: Low / Medium / High |
| `Air_temperature_K` | Ambient air temperature (Kelvin) |
| `Process_temperature_K` | Process/machine temperature (Kelvin) |
| `Rotational_speed_rpm` | Tool rotational speed (rpm) |
| `Torque_Nm` | Torque (Newton-meters) |
| `Tool_wear_min` | Cumulative tool wear (minutes) |
| `Machine_failure` | **Target**: 1 if any failure mode triggered |
| `Failure_mode` | Which mechanism caused the failure |

Five real-world failure mechanisms are simulated:

| Mode | Cause |
|---|---|
| **TWF** | Tool Wear Failure — tool wear exceeds a quality-dependent threshold |
| **HDF** | Heat Dissipation Failure — small air/process temp gap + low rotation speed |
| **PWF** | Power Failure — torque × rotational speed (power) too low or too high |
| **OSF** | Overstrain Failure — tool wear × torque exceeds a quality-dependent threshold |
| **RNF** | Random Failure — rare (0.3%), irreducible noise, mimics real unpredictable failures |

Class balance is realistic and imbalanced — **~11.9% failure rate** —
which is intentional: real industrial failure data is always heavily
skewed toward "healthy," and handling that imbalance correctly is a core
part of the exercise.

## 3. Project Structure

```
predictive_maintenance/
├── data/
│   └── sensor_data.csv          # generated dataset (10,000 rows)
├── models/
│   ├── best_model.joblib        # trained sklearn Pipeline (preprocessing + classifier)
│   └── feature_cols.joblib      # feature column order, needed for inference
├── outputs/
│   ├── 01_class_balance.png
│   ├── 02_failure_modes.png
│   ├── 03_feature_distributions.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_torque_vs_speed.png
│   ├── 06_roc_curves.png
│   ├── 07_pr_curves.png
│   ├── 08_confusion_matrix.png
│   ├── 09_feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.json
├── src/
│   ├── generate_data.py         # synthetic dataset generator
│   ├── eda.py                   # exploratory data analysis + plots
│   ├── train_model.py           # feature engineering + training + evaluation
│   └── predict.py               # inference on new sensor readings
├── requirements.txt
└── README.md
```

## 4. Methodology

### 4.1 Feature Engineering
Raw sensor readings alone correlate weakly with failure (see
`04_correlation_heatmap.png` — most correlations are near zero) because
real failures come from **interactions** between variables, not any single
sensor. Four physics-informed features were engineered to expose these
interactions to the model:

- `Temp_diff_K` = Process temp − Air temp (heat dissipation risk)
- `Power_W` = Torque × angular velocity (power failure risk)
- `Strain` = Tool wear × Torque (overstrain risk)
- `Wear_per_speed` = Tool wear / Rotational speed

### 4.2 Preprocessing
- Numeric features standardized with `StandardScaler`
- Categorical `Type` one-hot encoded
- Combined via `ColumnTransformer` inside an sklearn `Pipeline`, so the
  exact same transformation is applied at inference time — no train/serve
  skew.

### 4.3 Handling Class Imbalance
Since failures are only ~12% of the data, plain accuracy is misleading (a
model that always predicts "no failure" would score 88% and be useless).
Instead:
- `class_weight="balanced"` for Logistic Regression and Random Forest
- `scale_pos_weight` for XGBoost
- Evaluation focuses on **precision/recall/F1 for the failure class** and
  **ROC-AUC / PR-AUC**, not raw accuracy

### 4.4 Models Compared
Three classifiers were trained and compared on an 80/20 stratified
train/test split:

| Model | Precision (Failure) | Recall (Failure) | F1 (Failure) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.200 | 0.650 | 0.306 | 0.709 | 0.326 |
| **Random Forest** ⭐ | **0.412** | **0.599** | **0.488** | **0.835** | **0.401** |
| XGBoost | 0.408 | 0.582 | 0.480 | 0.821 | 0.400 |

**Random Forest was selected as the best model** (highest F1 on the
failure class and highest ROC-AUC). Logistic Regression, being a linear
model, struggles to capture the interaction effects that actually drive
failures — reinforcing why the engineered interaction features and a
non-linear model both matter here.

### 4.5 Feature Importance
The Random Forest confirms the engineering was worthwhile — `Power_W`
(the engineered power-failure signal) is the single most important
feature, followed by `Tool_wear_min` and `Torque_Nm` (see
`09_feature_importance.png`).

## 5. How to Run

```bash
pip install -r requirements.txt

cd src
python generate_data.py      # generates data/sensor_data.csv
python eda.py                 # generates EDA plots in outputs/
python train_model.py         # trains, compares, and saves the best model
python predict.py             # runs example predictions
```

### Using the model programmatically
```python
from predict import predict_failure

result = predict_failure(
    air_temp=300.0, process_temp=310.0,
    rot_speed=1400, torque=60, tool_wear=230,
    machine_type="L"
)
# {'prediction': 1, 'failure_probability': 0.56, 'risk_level': 'High'}
```

## 6. Limitations & Future Work

- **Synthetic data**: results demonstrate the pipeline and methodology
  correctly, but a real deployment should retrain on actual sensor logs
  from the target machines, since real-world noise and drift will differ.
- **Recall vs. precision trade-off**: current threshold (0.5) balances
  both; in a real safety-critical setting you'd likely lower the
  threshold to catch more failures at the cost of more false alarms — the
  PR curve (`07_pr_curves.png`) lets you pick that operating point.
- **Time-series structure**: this project treats each reading as
  independent. A production system would use sequential/windowed
  features (e.g., rate of change in tool wear over the last N readings)
  or a time-series model (LSTM, temporal CNN) for better early warning.
- **Multi-class extension**: `Failure_mode` labels are generated but not
  yet modeled directly — a natural next step is a multi-class classifier
  to predict *which* failure mode is most likely, not just whether one
  will occur.

## 7. Key Takeaways for the Report

1. Predictive maintenance is fundamentally an **imbalanced classification**
   problem — accuracy is the wrong metric; use precision/recall/F1/ROC-AUC.
2. **Domain-informed feature engineering** (interaction terms grounded in
   physics) matters more than model choice — it's why tree-based models
   beat linear models here.
3. Model selection should be driven by the **business cost of errors**: a
   missed failure (false negative) is typically far more expensive than a
   false alarm (false positive), which should inform the classification
   threshold in a real deployment.
