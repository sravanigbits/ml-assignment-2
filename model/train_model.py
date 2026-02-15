"""
ML Assignment 2 - Classification Models Training Script
========================================================
Dataset: Online Shoppers Purchasing Intention Dataset (UCI ML Repository)
Task: Binary Classification - Predict whether an online shopping session 
      will result in a purchase (Revenue: True/False)

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics:
- Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


def load_and_preprocess_data():
    """
    Load the Online Shoppers Purchasing Intention dataset and preprocess it.
    The dataset is available at:
    https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
    
    Features (18 total, 17 input + 1 target):
    - Administrative, Administrative_Duration, Informational, Informational_Duration
    - ProductRelated, ProductRelated_Duration
    - BounceRates, ExitRates, PageValues, SpecialDay
    - Month, OperatingSystems, Browser, Region, TrafficType
    - VisitorType, Weekend
    - Revenue (Target variable - True/False)
    """
    # Download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"Dataset loaded from UCI Repository successfully!")
    except Exception:
        # Fallback: try loading from local file
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "online_shoppers_intention.csv")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            print("Dataset loaded from local file.")
        else:
            print("Downloading dataset using alternative URL...")
            df = pd.read_csv(
                "https://raw.githubusercontent.com/dsrscientist/dataset1/master/online_shoppers_intention.csv"
            )
            print("Dataset loaded successfully!")

    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1] - 1}")
    print(f"Number of Instances: {df.shape[0]}")
    print(f"\nTarget Distribution:\n{df['Revenue'].value_counts()}")
    print(f"\nTarget Distribution (%):\n{df['Revenue'].value_counts(normalize=True) * 100}")

    # Handle missing values
    df = df.dropna()
    print(f"\nAfter dropping NaN - Shape: {df.shape}")

    # Encode categorical variables
    label_encoders = {}

    # Month encoding
    le_month = LabelEncoder()
    df["Month"] = le_month.fit_transform(df["Month"].astype(str))
    label_encoders["Month"] = le_month

    # VisitorType encoding
    le_visitor = LabelEncoder()
    df["VisitorType"] = le_visitor.fit_transform(df["VisitorType"].astype(str))
    label_encoders["VisitorType"] = le_visitor

    # Weekend encoding (boolean to int)
    df["Weekend"] = df["Weekend"].astype(int)

    # Revenue (target) encoding
    df["Revenue"] = df["Revenue"].astype(int)

    # Separate features and target
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # Feature names for later use
    feature_names = list(X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names, df


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model and return all required metrics.
    """
    y_pred = model.predict(X_test)

    # Get probability predictions for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall_val = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall_val, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4),
    }

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  AUC Score : {auc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall_val:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  MCC Score : {mcc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, cm, report


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 classification models and evaluate them.
    """
    models = {}
    all_metrics = []
    all_cm = {}
    all_reports = {}

    # ==========================================
    # 1. Logistic Regression
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: Logistic Regression")
    print("=" * 60)
    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs", C=1.0
    )
    lr_model.fit(X_train, y_train)
    models["Logistic Regression"] = lr_model
    metrics, cm, report = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    all_metrics.append(metrics)
    all_cm["Logistic Regression"] = cm
    all_reports["Logistic Regression"] = report

    # ==========================================
    # 2. Decision Tree Classifier
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: Decision Tree Classifier")
    print("=" * 60)
    dt_model = DecisionTreeClassifier(
        random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    dt_model.fit(X_train, y_train)
    models["Decision Tree"] = dt_model
    metrics, cm, report = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    all_metrics.append(metrics)
    all_cm["Decision Tree"] = cm
    all_reports["Decision Tree"] = report

    # ==========================================
    # 3. K-Nearest Neighbor Classifier
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: K-Nearest Neighbor Classifier")
    print("=" * 60)
    knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski")
    knn_model.fit(X_train, y_train)
    models["kNN"] = knn_model
    metrics, cm, report = evaluate_model(knn_model, X_test, y_test, "kNN")
    all_metrics.append(metrics)
    all_cm["kNN"] = cm
    all_reports["kNN"] = report

    # ==========================================
    # 4. Naive Bayes Classifier (Gaussian)
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: Gaussian Naive Bayes Classifier")
    print("=" * 60)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    models["Naive Bayes"] = nb_model
    metrics, cm, report = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    all_metrics.append(metrics)
    all_cm["Naive Bayes"] = cm
    all_reports["Naive Bayes"] = report

    # ==========================================
    # 5. Random Forest (Ensemble)
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: Random Forest (Ensemble)")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models["Random Forest (Ensemble)"] = rf_model
    metrics, cm, report = evaluate_model(rf_model, X_test, y_test, "Random Forest (Ensemble)")
    all_metrics.append(metrics)
    all_cm["Random Forest (Ensemble)"] = cm
    all_reports["Random Forest (Ensemble)"] = report

    # ==========================================
    # 6. XGBoost (Ensemble)
    # ==========================================
    print("\n" + "=" * 60)
    print("Training: XGBoost (Ensemble)")
    print("=" * 60)
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost (Ensemble)"] = xgb_model
    metrics, cm, report = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Ensemble)")
    all_metrics.append(metrics)
    all_cm["XGBoost (Ensemble)"] = cm
    all_reports["XGBoost (Ensemble)"] = report

    return models, all_metrics, all_cm, all_reports


def save_models_and_artifacts(models, scaler, label_encoders, feature_names, 
                               all_metrics, all_cm, all_reports, X_test, y_test):
    """
    Save all trained models and artifacts for the Streamlit app.
    """
    save_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Save each model
    for name, model in models.items():
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(save_dir, f"{filename}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {filepath}")

    # Save scaler
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Saved: scaler.pkl")

    # Save label encoders
    with open(os.path.join(save_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    print("Saved: label_encoders.pkl")

    # Save feature names
    with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print("Saved: feature_names.json")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(save_dir, "model_metrics.csv"), index=False)
    print("Saved: model_metrics.csv")

    # Save confusion matrices
    with open(os.path.join(save_dir, "confusion_matrices.pkl"), "wb") as f:
        pickle.dump(all_cm, f)
    print("Saved: confusion_matrices.pkl")

    # Save classification reports
    with open(os.path.join(save_dir, "classification_reports.pkl"), "wb") as f:
        pickle.dump(all_reports, f)
    print("Saved: classification_reports.pkl")

    # Save test data for demo purposes
    test_data = pd.DataFrame(X_test)
    test_data["Revenue"] = y_test.values
    test_data.to_csv(os.path.join(save_dir, "test_data.csv"), index=False)
    print("Saved: test_data.csv")


def print_comparison_table(all_metrics):
    """
    Print a formatted comparison table of all models.
    """
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index("Model")
    print(metrics_df.to_string())

    print("\n" + "=" * 90)
    print("BEST MODEL IDENTIFICATION")
    print("=" * 90)

    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]:
        best_model = metrics_df[metric].idxmax()
        best_value = metrics_df[metric].max()
        print(f"  Best {metric:12s}: {best_model:30s} ({best_value:.4f})")


def main():
    """
    Main function to run the complete ML pipeline.
    """
    print("=" * 60)
    print("ML ASSIGNMENT 2 - CLASSIFICATION MODELS")
    print("Online Shoppers Purchasing Intention Dataset")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names, df = (
        load_and_preprocess_data()
    )

    # Step 2: Train all models
    print("\n[Step 2] Training all 6 classification models...")
    models, all_metrics, all_cm, all_reports = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # Step 3: Print comparison table
    print("\n[Step 3] Model Comparison...")
    print_comparison_table(all_metrics)

    # Step 4: Save models and artifacts
    print("\n[Step 4] Saving models and artifacts...")
    save_models_and_artifacts(
        models, scaler, label_encoders, feature_names,
        all_metrics, all_cm, all_reports, X_test, y_test
    )

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    metrics = main()
