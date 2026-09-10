"""
generate_data.py
-----------------
Generates a synthetic industrial machine sensor dataset for the
Predictive Maintenance project.

The schema and failure logic are modeled on the well-known AI4I 2020
Predictive Maintenance Dataset (a standard benchmark in PdM literature),
so the project structure, feature set, and failure modes reflect a
realistic industrial setting:

Features
    UDI                    : unique row id
    Product_ID              : product type + serial (L/M/H quality variants)
    Type                    : Low / Medium / High quality variant
    Air_temperature_K        : ambient air temperature (Kelvin)
    Process_temperature_K    : process temperature (Kelvin)
    Rotational_speed_rpm     : tool rotational speed (rpm)
    Torque_Nm               : torque (Newton-meters)
    Tool_wear_min            : cumulative tool wear (minutes)

Target
    Machine_failure         : binary flag, 1 if ANY failure mode triggered
    Failure_mode             : which mechanism caused failure (or "None")

Failure modes simulated (matches real PdM taxonomy):
    TWF  - Tool Wear Failure
    HDF  - Heat Dissipation Failure
    PWF  - Power Failure
    OSF  - Overstrain Failure
    RNF  - Random failure (unpredictable, small noise term)

Run:
    python generate_data.py --n 10000 --seed 42 --out ../data/sensor_data.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Product type / quality variant ----------------------------------
    # L = Low quality (50%), M = Medium (30%), H = High (20%) -- mirrors
    # typical industrial batch mix.
    type_choices = rng.choice(["L", "M", "H"], size=n_samples, p=[0.5, 0.3, 0.2])
    type_offset = {"L": 0.0, "M": 1.0, "H": 2.0}  # higher quality -> slightly sturdier

    # ---- Base sensor readings ---------------------------------------------
    air_temp = rng.normal(loc=300.0, scale=2.0, size=n_samples)  # Kelvin
    process_temp = air_temp + rng.normal(loc=10.0, scale=1.0, size=n_samples)

    rotational_speed = rng.normal(loc=1500, scale=180, size=n_samples)
    rotational_speed = np.clip(rotational_speed, 900, 2900)

    torque = rng.normal(loc=40, scale=10, size=n_samples)
    torque = np.clip(torque, 3, 80)

    tool_wear = rng.uniform(0, 253, size=n_samples)

    quality_bonus = np.array([type_offset[t] for t in type_choices])

    # ---- Failure mode logic (probabilistic, threshold-driven) --------------
    # Each rule below approximates the physical conditions under which a
    # real machine tends to fail, then adds randomness so it isn't a
    # trivial rule-based dataset (keeps the ML problem meaningful).

    # Tool Wear Failure: high wear + randomness, less likely for higher quality tools
    twf_prob = np.clip((tool_wear - (200 - quality_bonus * 5)) / 60, 0, 1) * 0.35
    twf = rng.random(n_samples) < twf_prob

    # Heat Dissipation Failure: small temp diff + low rotational speed
    temp_diff = process_temp - air_temp
    hdf_cond = (temp_diff < 8.6) & (rotational_speed < 1380)
    hdf_prob = np.where(hdf_cond, 0.55, 0.01)
    hdf = rng.random(n_samples) < hdf_prob

    # Power Failure: power = torque * rotational_speed (rad/s); too low or too high is bad
    power = torque * (rotational_speed * 2 * np.pi / 60)
    pwf_cond = (power < 3500) | (power > 9000)
    pwf_prob = np.where(pwf_cond, 0.4, 0.01)
    pwf = rng.random(n_samples) < pwf_prob

    # Overstrain Failure: tool_wear * torque exceeds a quality-dependent threshold
    strain = tool_wear * torque
    osf_threshold = 11000 + quality_bonus * 1500  # L=11000, M=12500, H=14000
    osf_prob = np.clip((strain - osf_threshold) / 4000, 0, 1) * 0.5
    osf = rng.random(n_samples) < osf_prob

    # Random failures: rare, unrelated to sensor readings (irreducible noise)
    rnf = rng.random(n_samples) < 0.003

    failure_flags = np.vstack([twf, hdf, pwf, osf, rnf]).T
    machine_failure = failure_flags.any(axis=1).astype(int)

    mode_names = np.array(["TWF", "HDF", "PWF", "OSF", "RNF"])

    def pick_mode(row):
        idx = np.where(row)[0]
        if len(idx) == 0:
            return "No_Failure"
        return mode_names[idx[0]]  # first triggered mode (rows can have >1 in reality)

    failure_mode = np.apply_along_axis(pick_mode, 1, failure_flags)

    df = pd.DataFrame({
        "UDI": np.arange(1, n_samples + 1),
        "Product_ID": [f"{t}{47000 + i}" for i, t in enumerate(type_choices)],
        "Type": type_choices,
        "Air_temperature_K": np.round(air_temp, 2),
        "Process_temperature_K": np.round(process_temp, 2),
        "Rotational_speed_rpm": np.round(rotational_speed, 0).astype(int),
        "Torque_Nm": np.round(torque, 2),
        "Tool_wear_min": np.round(tool_wear, 1),
        "TWF": twf.astype(int),
        "HDF": hdf.astype(int),
        "PWF": pwf.astype(int),
        "OSF": osf.astype(int),
        "RNF": rnf.astype(int),
        "Machine_failure": machine_failure,
        "Failure_mode": failure_mode,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic predictive maintenance dataset")
    parser.add_argument("--n", type=int, default=10000, help="number of samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default="../data/sensor_data.csv", help="output CSV path")
    args = parser.parse_args()

    data = generate_dataset(args.n, args.seed)
    data.to_csv(args.out, index=False)

    print(f"Generated {len(data)} rows -> {args.out}")
    print(f"Failure rate: {data['Machine_failure'].mean():.2%}")
    print(data["Failure_mode"].value_counts())
