import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score

st.set_page_config(page_title="AutoML Pipeline Builder", layout="wide")
st.title("AutoML Pipeline Builder Demo 🚀")

# -----------------------------
# Step 1: Upload dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Get file extension
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Read file based on extension
    try:
        if ext == ".csv":
            data = pd.read_csv(uploaded_file)
        elif ext in [".xls", ".xlsx"]:
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format! Only CSV and Excel files are allowed.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Step 2: Select target column
    target_col = st.selectbox("Select Target Column", data.columns)

    if st.button("Run AutoML Pipeline"):
        st.info("Loading the saved pipeline...")

        # Load the best saved model pipeline
        try:
            pipeline = joblib.load("artifacts/best_model.pkl")
        except:
            st.error("Best model not found. Please run main.py first.")
            st.stop()

        # Split features and target
        X = data.drop(columns=[target_col])
        y_true = data[target_col]

        # Make predictions
        y_pred = pipeline.predict(X)

        # Display predictions
        st.subheader("Predictions")
        predictions_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred
        })
        st.dataframe(predictions_df)

        # Download button for predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            "Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Display metrics
        st.subheader("Evaluation Metrics")
        try:
            # Determine if classification or regression
            is_classification = y_true.dtype == 'O' or y_true.nunique() <= 10

            if is_classification:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                st.write(f"**Accuracy:** {acc:.3f}")
                st.write(f"**F1 Score:** {f1:.3f}")
            else:
                r2 = r2_score(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                mae = mean_absolute_error(y_true, y_pred)
                st.write(f"**R² Score:** {r2:.3f}")
                st.write(f"**RMSE:** {rmse:.3f}")
                st.write(f"**MAE:** {mae:.3f}")
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")
