"""
predict.py
----------
Load the trained model and make failure predictions on new machine
sensor readings.

Run (uses built-in example readings):
    python predict.py

Or import and use programmatically:
    from predict import predict_failure
    result = predict_failure(air_temp=298.5, process_temp=308.9,
                              rot_speed=1500, torque=45.0,
                              tool_wear=200, machine_type="M")
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../models/best_model.joblib"
FEATURES_PATH = "../models/feature_cols.joblib"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_W"] = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)
    df["Strain"] = df["Tool_wear_min"] * df["Torque_Nm"]
    df["Wear_per_speed"] = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1)
    return df


def predict_failure(air_temp, process_temp, rot_speed, torque, tool_wear, machine_type="M"):
    """Predict failure probability for a single machine reading.

    Returns a dict with prediction (0/1), probability, and risk level.
    """
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    raw = pd.DataFrame([{
        "Type": machine_type,
        "Air_temperature_K": air_temp,
        "Process_temperature_K": process_temp,
        "Rotational_speed_rpm": rot_speed,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear,
    }])
    engineered = engineer_features(raw)
    X = engineered[feature_cols]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]

    if proba < 0.2:
        risk = "Low"
    elif proba < 0.5:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": int(pred),
        "failure_probability": round(float(proba), 4),
        "risk_level": risk,
    }


if __name__ == "__main__":
    examples = [
        # A healthy-looking machine
        dict(air_temp=298.5, process_temp=308.5, rot_speed=1550, torque=38, tool_wear=50, machine_type="H"),
        # A machine with high tool wear + high torque (overstrain risk)
        dict(air_temp=300.0, process_temp=310.0, rot_speed=1400, torque=60, tool_wear=230, machine_type="L"),
        # A machine with low temp differential + low speed (heat dissipation risk)
        dict(air_temp=302.0, process_temp=308.0, rot_speed=1300, torque=45, tool_wear=100, machine_type="M"),
    ]

    print("Predictive Maintenance — Example Inference\n" + "=" * 45)
    for i, ex in enumerate(examples, 1):
        result = predict_failure(**ex)
        print(f"\nMachine reading #{i}: {ex}")
        print(f"  -> Prediction: {'FAILURE RISK' if result['prediction'] else 'Normal'}")
        print(f"  -> Failure probability: {result['failure_probability']:.2%}")
        print(f"  -> Risk level: {result['risk_level']}")
