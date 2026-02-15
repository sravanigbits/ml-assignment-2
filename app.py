"""
ML Assignment 2 - Streamlit Web Application
=============================================
Interactive web application for demonstrating 6 ML classification models
on the Online Shoppers Purchasing Intention Dataset.

Features:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="ML Classification Models - Assignment 2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# Custom CSS Styling
# ==================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Helper Functions
# ==================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def get_model_path(filename):
    return os.path.join(MODEL_DIR, filename)


@st.cache_data
def load_default_dataset():
    """Load the Online Shoppers Purchasing Intention dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        try:
            df = pd.read_csv(
                "https://raw.githubusercontent.com/dsrscientist/dataset1/master/online_shoppers_intention.csv"
            )
        except Exception:
            st.error("Could not load the default dataset. Please upload a CSV file.")
            return None
    return df


def preprocess_data(df):
    """Preprocess the dataset for model training/evaluation."""
    df = df.dropna()

    # Encode categorical variables
    if "Month" in df.columns:
        le_month = LabelEncoder()
        df["Month"] = le_month.fit_transform(df["Month"].astype(str))

    if "VisitorType" in df.columns:
        le_visitor = LabelEncoder()
        df["VisitorType"] = le_visitor.fit_transform(df["VisitorType"].astype(str))

    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].astype(int)

    return df


def train_and_evaluate_models(df):
    """Train all 6 models and return metrics, models, confusion matrices."""
    df_processed = preprocess_data(df.copy())

    X = df_processed.drop("Revenue", axis=1)
    y = df_processed["Revenue"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost (Ensemble)": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric="logloss"),
    }

    all_metrics = []
    trained_models = {}
    all_cm = {}
    all_reports = {}

    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        }

        all_metrics.append(metrics)
        trained_models[name] = model
        all_cm[name] = confusion_matrix(y_test, y_pred)
        all_reports[name] = classification_report(y_test, y_pred, output_dict=True)

    return all_metrics, trained_models, all_cm, all_reports, X_test, y_test


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix using matplotlib/seaborn."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_df):
    """Create a bar chart comparing metrics across models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics_to_plot = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(metrics_df["Model"], metrics_df[metric], color=color, alpha=0.8, edgecolor="black")
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(metrics_df["Model"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==================================================
# Main Application
# ==================================================
def main():
    # Header
    st.markdown('<p class="main-header">🤖 ML Classification Models Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Assignment 2 - Online Shoppers Purchasing Intention Prediction<br>'
        "Comparing 6 Machine Learning Classification Models</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ==================================================
    # Sidebar
    # ==================================================
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset Upload
        st.subheader("📁 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV test data",
            type=["csv"],
            help="Upload a CSV file with the same features as the Online Shoppers dataset. "
                 "For Streamlit free tier, upload only test data.",
        )

        st.divider()

        # Model Selection
        st.subheader("🔧 Model Selection")
        model_names = [
            "All Models",
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)",
        ]
        selected_model = st.selectbox(
            "Choose a model to view details:",
            model_names,
            index=0,
            help="Select a specific model to view its detailed metrics, or 'All Models' for comparison.",
        )

        st.divider()

        # Dataset Info
        st.subheader("📊 Dataset Info")
        st.info(
            "**Dataset:** Online Shoppers Purchasing Intention\n\n"
            "**Source:** UCI ML Repository\n\n"
            "**Instances:** 12,330\n\n"
            "**Features:** 17\n\n"
            "**Target:** Revenue (Binary)\n\n"
            "**Task:** Binary Classification"
        )

    # ==================================================
    # Load and Process Data
    # ==================================================
    with st.spinner("Loading dataset and training models... This may take a moment."):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded dataset loaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                df = load_default_dataset()
        else:
            df = load_default_dataset()

    if df is None:
        st.error("Failed to load dataset. Please upload a valid CSV file.")
        return

    # ==================================================
    # Dataset Overview Tab
    # ==================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dataset Overview", "📈 Model Comparison", "🔍 Individual Model Details", "📋 Classification Reports"]
    )

    # Train models
    with st.spinner("Training all 6 classification models..."):
        all_metrics, trained_models, all_cm, all_reports, X_test, y_test = train_and_evaluate_models(df)

    metrics_df = pd.DataFrame(all_metrics)

    # ==================================================
    # Tab 1: Dataset Overview
    # ==================================================
    with tab1:
        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Instances", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1] - 1)
        with col3:
            if "Revenue" in df.columns:
                positive = df["Revenue"].sum() if df["Revenue"].dtype != object else (df["Revenue"] == True).sum()
                st.metric("Positive Class", int(positive))
            else:
                st.metric("Positive Class", "N/A")
        with col4:
            if "Revenue" in df.columns:
                negative = len(df) - int(positive)
                st.metric("Negative Class", negative)
            else:
                st.metric("Negative Class", "N/A")

        st.subheader("First 10 Rows of the Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        with col_b:
            st.subheader("Target Distribution")
            if "Revenue" in df.columns:
                fig_target, ax_target = plt.subplots(figsize=(6, 4))
                target_counts = df["Revenue"].value_counts()
                ax_target.bar(
                    ["No Purchase (False)", "Purchase (True)"],
                    target_counts.values,
                    color=["#ff7f0e", "#1f77b4"],
                    edgecolor="black",
                )
                ax_target.set_ylabel("Count")
                ax_target.set_title("Target Variable Distribution")
                for i, v in enumerate(target_counts.values):
                    ax_target.text(i, v + 50, str(v), ha="center", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_target)
                plt.close()

        st.subheader("Feature Information")
        feature_info = pd.DataFrame(
            {
                "Feature": df.columns,
                "Data Type": df.dtypes.values,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
                "Unique Values": df.nunique().values,
            }
        )
        st.dataframe(feature_info, use_container_width=True)

    # ==================================================
    # Tab 2: Model Comparison
    # ==================================================
    with tab2:
        st.header("📈 Model Comparison")

        # Metrics Table
        st.subheader("Evaluation Metrics Comparison Table")
        styled_metrics = metrics_df.set_index("Model")

        # Highlight best values
        st.dataframe(
            styled_metrics.style.highlight_max(axis=0, color="#90EE90").format("{:.4f}"),
            use_container_width=True,
        )

        # Best Model Summary
        st.subheader("🏆 Best Model per Metric")
        best_models = {}
        for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]:
            best_idx = metrics_df[metric].idxmax()
            best_model_name = metrics_df.loc[best_idx, "Model"]
            best_value = metrics_df.loc[best_idx, metric]
            best_models[metric] = f"{best_model_name} ({best_value:.4f})"

        cols = st.columns(3)
        for idx, (metric, value) in enumerate(best_models.items()):
            with cols[idx % 3]:
                st.success(f"**{metric}:** {value}")

        # Visual Comparison
        st.subheader("Visual Comparison of All Models")
        fig_comparison = plot_metrics_comparison(metrics_df)
        st.pyplot(fig_comparison)
        plt.close()

        # Radar Chart
        st.subheader("Radar Chart - Model Comparison")
        fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        metrics_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        angles = np.linspace(0, 2 * np.pi, len(metrics_cols), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors_radar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, row in metrics_df.iterrows():
            values = [row[m] for m in metrics_cols]
            values += values[:1]
            ax_radar.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=colors_radar[idx])
            ax_radar.fill(angles, values, alpha=0.1, color=colors_radar[idx])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_cols, fontsize=10)
        ax_radar.set_ylim(0, 1.0)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax_radar.set_title("Model Performance Radar Chart", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig_radar)
        plt.close()

    # ==================================================
    # Tab 3: Individual Model Details
    # ==================================================
    with tab3:
        st.header("🔍 Individual Model Details")

        if selected_model == "All Models":
            st.info("Select a specific model from the sidebar dropdown to view detailed results.")

            # Show all confusion matrices
            st.subheader("Confusion Matrices - All Models")
            cols_cm = st.columns(3)
            for idx, (name, cm) in enumerate(all_cm.items()):
                with cols_cm[idx % 3]:
                    fig_cm = plot_confusion_matrix(cm, name)
                    st.pyplot(fig_cm)
                    plt.close()
        else:
            st.subheader(f"📊 {selected_model}")

            # Metrics
            model_metrics = metrics_df[metrics_df["Model"] == selected_model].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
                st.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
            with col2:
                st.metric("Precision", f"{model_metrics['Precision']:.4f}")
                st.metric("Recall", f"{model_metrics['Recall']:.4f}")
            with col3:
                st.metric("F1 Score", f"{model_metrics['F1 Score']:.4f}")
                st.metric("MCC Score", f"{model_metrics['MCC']:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = all_cm[selected_model]
            fig_cm = plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig_cm)
            plt.close()

            # Detailed Metrics
            st.subheader("Detailed Metrics")
            metrics_detail = {
                "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"],
                "Value": [
                    model_metrics["Accuracy"],
                    model_metrics["AUC"],
                    model_metrics["Precision"],
                    model_metrics["Recall"],
                    model_metrics["F1 Score"],
                    model_metrics["MCC"],
                ],
            }
            st.dataframe(pd.DataFrame(metrics_detail), use_container_width=True)

    # ==================================================
    # Tab 4: Classification Reports
    # ==================================================
    with tab4:
        st.header("📋 Classification Reports")

        if selected_model == "All Models":
            for name, report in all_reports.items():
                st.subheader(f"📄 {name}")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                st.divider()
        else:
            st.subheader(f"📄 Classification Report - {selected_model}")
            report = all_reports[selected_model]
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # ==================================================
    # Footer
    # ==================================================
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #888; padding: 20px;">
            <p>ML Assignment 2 | M.Tech AIML/DSE | BITS Pilani - WILP</p>
            <p>Online Shoppers Purchasing Intention Dataset | UCI ML Repository</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
