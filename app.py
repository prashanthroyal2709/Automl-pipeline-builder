import streamlit as st
import pandas as pd
from main import run_automl

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AutoML Pipeline Builder",
    layout="centered"
)

st.title("🚀 AutoML Pipeline Builder")
st.write(
    "Upload a CSV file, select the target column, and automatically "
    "train and evaluate the best machine learning model."
)

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully ✅")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Target column selection
    # -----------------------------
    target_col = st.selectbox(
        "Select the Target Column",
        df.columns
    )

    # -----------------------------
    # Run AutoML
    # -----------------------------
    if st.button("Run AutoML 🚀"):
        with st.spinner("Training models and selecting the best one..."):
            best_model, best_score, metrics_df = run_automl(df, target_col)

        st.success("AutoML Completed Successfully 🎉")

        st.subheader("Best Model Performance")
        st.metric(
            label="Best Score",
            value=round(best_score, 4)
        )

        st.subheader("Model Comparison")
        st.dataframe(
            metrics_df[["model_name", "score"]]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

else:
    st.info("👆 Please upload a CSV file to get started")
