import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hacker News Popularity Prediction",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

DATA_FILE = Path(__file__).parent / "hacker_news_model_data.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# Check dataset
# ---------------------------------------------------------

if not DATA_FILE.exists():
    st.error("Dataset file not found.")
    st.write("Expected file:")
    st.code("hacker_news_model_data.csv")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error("The dataset could not be loaded.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📰 Hacker News Post Popularity Prediction")

st.write(
    "Interactive dashboard for analyzing Hacker News post popularity "
    "using machine learning."
)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Analysis",
        "Model Performance",
        "Responsible AI",
        "Final Portfolio"
    ]
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Posts",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Model Features",
            "24"
        )

    with col3:
        st.metric(
            "Popularity Classes",
            "3"
        )

    st.divider()

    st.subheader("Problem Statement")

    st.write(
        "The objective of this project is to predict the popularity of "
        "Hacker News posts using machine learning. Posts are classified "
        "into three popularity categories: Low, Medium, and High."
    )

    st.subheader("Popularity Classes")

    class_data = pd.DataFrame(
        {
            "Class": ["Low", "Medium", "High"],
            "Class Code": [0, 1, 2],
            "Description": [
                "Low popularity",
                "Medium popularity",
                "High popularity"
            ]
        }
    )

    st.dataframe(
        class_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Project Pipeline")

    st.write(
        """
        **Experiment 1:** Case Study Framing & Dataset Preparation

        **Experiment 2:** Data Profiling, Cleaning & Feature Engineering

        **Experiment 3:** EDA & Statistical Analysis

        **Experiment 4:** ML Modeling & Experiment Tracking

        **Experiment 5:** Explainable AI & Fairness Evaluation

        **Experiment 6:** Containerization & API Deployment

        **Experiment 7:** CI/CD Pipeline

        **Experiment 8:** Dashboard, Responsible AI Reporting & Final Portfolio
        """
    )


# =========================================================
# DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.header("Dataset Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Records", f"{len(df):,}")

    with col2:
        st.metric("Columns", f"{len(df.columns):,}")

    with col3:
        st.metric(
            "Missing Values",
            f"{int(df.isnull().sum().sum()):,}"
        )

    st.divider()

    # Popularity distribution

    st.subheader("Popularity Distribution")

    if "popularity_label" in df.columns:

        class_counts = (
            df["popularity_label"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .astype(int)
        )

        distribution_df = pd.DataFrame(
            {
                "Popularity Class": class_counts.index,
                "Number of Posts": class_counts.values
            }
        )

        st.bar_chart(
            distribution_df.set_index("Popularity Class")
        )

        st.dataframe(
            distribution_df,
            use_container_width=True,
            hide_index=True
        )

    # Dataset information

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Dataset Shape**")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]:,}")

    with info_col2:
        st.write("**Data Quality**")
        st.write(
            f"Missing values: {int(df.isnull().sum().sum()):,}"
        )
        st.write(
            f"Duplicate rows: {int(df.duplicated().sum()):,}"
        )

    st.divider()

    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Numerical summary

    st.subheader("Numerical Feature Summary")

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_columns) > 0:
        st.dataframe(
            df[numeric_columns].describe().round(2),
            use_container_width=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "The following results were obtained during Experiment 4 "
        "using the 600-post test set."
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Gradient Boosting",
                "Tuned Random Forest",
                "Tuned Gradient Boosting"
            ],
            "Accuracy": [
                0.4500,
                0.3983,
                0.4567,
                0.4017,
                0.4617,
                0.4617,
                0.4533
            ],
            "Macro F1": [
                0.3886,
                0.3908,
                0.4184,
                0.1910,
                0.4184,
                0.4181,
                0.3961
            ]
        }
    )

    st.subheader("Model Comparison")

    st.dataframe(
        results.style.format(
            {
                "Accuracy": "{:.4f}",
                "Macro F1": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Comparison")

    accuracy_chart = results.set_index("Model")[["Accuracy"]]

    st.bar_chart(accuracy_chart)

    st.subheader("Macro F1 Comparison")

    f1_chart = results.set_index("Model")[["Macro F1"]]

    st.bar_chart(f1_chart)

    st.divider()

    st.subheader("Selected Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "Gradient Boosting"
        )

    with col2:
        st.metric(
            "Accuracy",
            "46.17%"
        )

    with col3:
        st.metric(
            "Macro F1",
            "0.4184"
        )

    st.write(
        "The baseline Gradient Boosting model was selected as the final "
        "model because it achieved an accuracy of approximately 0.4617 "
        "and a macro F1 score of approximately 0.4184."
    )

    st.subheader("Class-Level Performance")

    classification_report = pd.DataFrame(
        {
            "Class": ["Low", "Medium", "High"],
            "Precision": [0.47, 0.44, 0.45],
            "Recall": [0.73, 0.22, 0.34],
            "F1 Score": [0.57, 0.29, 0.39],
            "Support": [241, 184, 175]
        }
    )

    st.dataframe(
        classification_report.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1 Score": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# RESPONSIBLE AI
# =========================================================

elif page == "Responsible AI":

    st.header("Responsible AI Evaluation")

    st.subheader("1. Target Leakage")

    st.write(
        "Variables that could directly reveal the popularity target "
        "were excluded from the model input. In particular, score and "
        "engagement-related variables were not used as predictive "
        "features where they could cause target leakage."
    )

    st.subheader("2. Explainable AI")

    st.write(
        "SHAP was used to analyze the contribution of model features "
        "to predictions. A model-agnostic SHAP explainer was used "
        "because the multiclass Gradient Boosting model was not "
        "compatible with the TreeExplainer approach used initially."
    )

    st.subheader("3. Fairness Evaluation")

    st.write(
        "Fairness evaluation was attempted using available subgroup "
        "information. The available test data contained insufficient "
        "variation for meaningful subgroup comparisons in some cases."
    )

    st.write(
        "For example, the verified post-type evaluation contained "
        "only the story category, while the weekend comparison did "
        "not contain a weekend subgroup. Therefore, these results "
        "should not be interpreted as evidence that the model is "
        "universally fair or unfair."
    )

    st.subheader("4. Model Limitations")

    limitations = [
        "The dataset contains 3,000 Hacker News posts.",
        "Popularity can be affected by factors that are not captured by the available features.",
        "The model does not guarantee future popularity.",
        "Performance differs across the Low, Medium, and High classes.",
        "The fairness analysis is limited by the available subgroup data."
    ]

    for item in limitations:
        st.write("• " + item)

    st.subheader("5. Responsible AI Summary")

    st.info(
        "The model should be treated as an analytical prediction tool "
        "rather than a guarantee of future post popularity. Its "
        "predictions should be interpreted together with the dataset "
        "limitations and class-level performance."
    )


# =========================================================
# FINAL PORTFOLIO
# =========================================================

elif page == "Final Portfolio":

    st.header("Final Project Portfolio")

    st.subheader("Project Title")

    st.write(
        "Hacker News Post Popularity Prediction using Machine Learning"
    )

    st.subheader("Completed Experiments")

    experiments = [
        "Experiment 1 — Case Study Framing & Dataset Preparation",
        "Experiment 2 — Data Profiling, Cleaning & Feature Engineering",
        "Experiment 3 — EDA & Statistical Analysis",
        "Experiment 4 — ML Modeling & Experiment Tracking",
        "Experiment 5 — Explainable AI & Fairness Evaluation",
        "Experiment 6 — Containerization & API Deployment",
        "Experiment 7 — CI/CD Pipeline",
        "Experiment 8 — Dashboard, Responsible AI Reporting & Final Portfolio"
    ]

    for experiment in experiments:
        st.write("✅ " + experiment)

    st.divider()

    st.subheader("Deployment Components")

    deployment_components = [
        "FastAPI prediction API",
        "Saved Gradient Boosting model",
        "Dockerfile",
        "GitHub repository",
        "GitHub Actions CI/CD workflow",
        "Streamlit interactive dashboard"
    ]

    for component in deployment_components:
        st.write("• " + component)

    st.divider()

    st.subheader("Final Model")

    final_model = pd.DataFrame(
        {
            "Metric": [
                "Model",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1"
            ],
            "Result": [
                "Gradient Boosting",
                "0.4617",
                "0.4529",
                "0.4315",
                "0.4184"
            ]
        }
    )

    st.dataframe(
        final_model,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Hacker News Popularity Prediction project portfolio completed."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Hacker News Popularity Prediction • Machine Learning & "
    "Explainable AI Project"
)
