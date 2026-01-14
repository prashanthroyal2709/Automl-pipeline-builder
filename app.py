import streamlit as st
import pandas as pd

st.write("HELLO STREAMLIT")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

# 🔽 ADD THIS
import data_ingestion
st.write("data_ingestion imported successfully")
