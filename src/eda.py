"""
eda.py
------
Exploratory Data Analysis for the predictive maintenance dataset.
Generates summary statistics and saves visualizations to ../outputs/.

Run:
    python eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

DATA_PATH = "../data/sensor_data.csv"
OUT_DIR = "../outputs"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
    print(f"\nClass balance (Machine_failure):\n{df['Machine_failure'].value_counts(normalize=True)}")

    numeric_cols = [
        "Air_temperature_K", "Process_temperature_K",
        "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"
    ]

    print(f"\nDescriptive statistics:\n{df[numeric_cols].describe()}")

    # 1. Class balance plot
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Machine_failure"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["No Failure", "Failure"], rotation=0)
    ax.set_title("Class Balance: Machine Failure")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_class_balance.png", dpi=150)
    plt.close()

    # 2. Failure mode breakdown
    fig, ax = plt.subplots(figsize=(6, 4))
    df[df.Machine_failure == 1]["Failure_mode"].value_counts().plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_title("Failure Mode Breakdown (failures only)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_failure_modes.png", dpi=150)
    plt.close()

    # 3. Feature distributions by failure status
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.kdeplot(data=df, x=col, hue="Machine_failure", fill=True, alpha=0.4, ax=axes[i])
        axes[i].set_title(col)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_feature_distributions.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols + ["Machine_failure"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()

    # 5. Torque vs rotational speed scatter, colored by failure
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="Rotational_speed_rpm", y="Torque_Nm",
        hue="Machine_failure", alpha=0.5, palette=["#4C72B0", "#C44E52"], ax=ax
    )
    ax.set_title("Torque vs Rotational Speed (colored by failure)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_torque_vs_speed.png", dpi=150)
    plt.close()

    print(f"\nSaved 5 plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
