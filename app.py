import streamlit as st
import pandas as pd
import joblib
import io

from main import run_automl

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AutoML Pipeline Builder",
    layout="centered"
)

st.title("🚀 AutoML Pipeline Builder")
st.write(
    "Upload a CSV file, select the target column, and automatically "
    "train, evaluate, and download the best machine learning model."
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
            best_model, best_model_name, best_score, metrics_df = run_automl(
                df, target_col
            )

        st.success("AutoML Completed Successfully 🎉")

        # -----------------------------
        # Best model details
        # -----------------------------
        st.subheader("Best Model Details")
        st.write(f"🏆 **Best Model:** `{best_model_name}`")
        st.metric("Best Score", round(best_score, 4))

        # -----------------------------
        # Model comparison table
        # -----------------------------
        st.subheader("Model Comparison")
        st.dataframe(
            metrics_df[["model_name", "score"]]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        # -----------------------------
        # Download trained model
        # -----------------------------
        buffer = io.BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download Best Model (.pkl)",
            data=buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

else:
    st.info("👆 Please upload a CSV file to get started")
