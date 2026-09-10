"""
train_model.py
---------------
Trains and compares multiple classifiers for predictive maintenance
(binary failure prediction), with engineered physics-informed features,
class-imbalance handling, and full evaluation.

Run:
    python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, average_precision_score
)
from xgboost import XGBClassifier

DATA_PATH = "../data/sensor_data.csv"
MODEL_DIR = "../models"
OUT_DIR = "../outputs"
RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed engineered features that help models
    capture the interaction effects behind real failure mechanisms."""
    df = df.copy()
    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_W"] = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)
    df["Strain"] = df["Tool_wear_min"] * df["Torque_Nm"]
    df["Wear_per_speed"] = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1)
    return df


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = [
        "Type", "Air_temperature_K", "Process_temperature_K",
        "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
        "Temp_diff_K", "Power_W", "Strain", "Wear_per_speed"
    ]
    X = df[feature_cols]
    y = df["Machine_failure"]

    categorical = ["Type"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical),
    ])

    return X, y, preprocessor, feature_cols


def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "precision_failure": report["1"]["precision"],
        "recall_failure": report["1"]["recall"],
        "f1_failure": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")

    return y_pred, y_proba


def main():
    X, y, preprocessor, feature_cols = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train failure rate: {y_train.mean():.2%} | Test failure rate: {y_test.mean():.2%}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE, eval_metric="logloss"
        ),
    }

    results = {}
    fitted_pipelines = {}
    roc_data = {}
    pr_data = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred, y_proba = evaluate_model(name, pipe, X_test, y_test, results)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = (prec, rec)

    # ---- Pick best model by F1 on the failure class (most relevant for PdM) ----
    best_name = max(results, key=lambda k: results[k]["f1_failure"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n*** Best model: {best_name} (F1-failure = {results[best_name]['f1_failure']:.4f}) ***")

    # Save the best model
    joblib.dump(best_pipe, f"{MODEL_DIR}/best_model.joblib")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.joblib")

    # Save metrics comparison
    with open(f"{OUT_DIR}/model_comparison.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # ---- Plots ----
    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_roc_curves.png", dpi=150)
    plt.close()

    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (prec, rec) in pr_data.items():
        ax.plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_pr_curves.png", dpi=150)
    plt.close()

    # Confusion matrix for best model
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (if tree-based model won)
    classifier = best_pipe.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        ohe_cols = list(best_pipe.named_steps["preprocess"]
                         .named_transformers_["cat"].get_feature_names_out(["Type"]))
        numeric_cols = [c for c in feature_cols if c != "Type"]
        all_cols = numeric_cols + ohe_cols

        importances = classifier.feature_importances_
        imp_df = pd.DataFrame({"feature": all_cols, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="#4C72B0")
        ax.set_title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/09_feature_importance.png", dpi=150)
        plt.close()

        imp_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    print(f"\nModel + metrics + plots saved to {MODEL_DIR}/ and {OUT_DIR}/")


if __name__ == "__main__":
    main()
