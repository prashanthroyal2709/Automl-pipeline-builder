import streamlit as st
import pandas as pd
import os
import joblib

from main import main  # Only if you want to run full pipeline dynamically

st.set_page_config(page_title="AutoML Pipeline Builder", layout="wide")

st.title("AutoML Pipeline Builder Demo 🚀")
st.write("Upload your dataset (CSV or Excel) and select the target column.")

# -----------------------------
# Step 1: Upload dataset
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    # Determine file extension
    ext = uploaded_file.name.split(".")[-1]
    if ext.lower() == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Step 2: Select target column
    target_col = st.selectbox("Select Target Column", df.columns)

    # Step 3: Load saved pipeline
    pipeline_path = "artifacts/best_model.pkl"
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
        st.success("Loaded saved pipeline from artifacts/best_model.pkl ✅")
    else:
        st.warning("Best model not found. Running main.py to train pipeline...")
        # Save uploaded dataset temporarily to pass to main.py
        temp_path = "temp_dataset.csv"
        df.to_csv(temp_path, index=False)
        # Run your main.py pipeline
        os.system(f'python main.py')  # Optional: trains pipeline
        if os.path.exists(pipeline_path):
            pipeline = joblib.load(pipeline_path)
            st.success("Pipeline trained and loaded successfully ✅")
        else:
            st.error("Pipeline could not be trained. Check main.py.")
            st.stop()

    # Step 4: Make predictions
    X = df.drop(columns=[target_col])
    y_true = df[target_col] if target_col in df else None

    st.subheader("Predictions")
    try:
        predictions = pipeline.predict(X)
        df_pred = pd.DataFrame({"Predictions": predictions})
        st.dataframe(df_pred.head())
    except Exception as e:
        st.error(f"Prediction failed: {e}")
