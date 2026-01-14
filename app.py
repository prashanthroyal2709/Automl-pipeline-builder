import streamlit as st
import pandas as pd
from main import run_automl

st.set_page_config(page_title="AutoML Pipeline Builder")

st.title("🚀 AutoML Pipeline Builder")
st.write(
    "Upload a CSV file, select the target column, and automatically "
    "train and evaluate the best machine learning model."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the Target Column", df.columns)

    if st.button("Run AutoML"):
        with st.spinner("Training models..."):
            best_model, best_model_name, best_score, metrics_df = run_automl(
                df, target_col
            )

        st.success("AutoML Completed 🎉")

        st.subheader("Best Model Details")
        st.write(f"🏆 **Best Model:** `{best_model_name}`")
        st.metric("Best Score", round(best_score, 4))

        st.subheader("Model Comparison")
        st.dataframe(
            metrics_df[["model_name", "score"]]
            .sort_values("score", ascending=False)
        )
