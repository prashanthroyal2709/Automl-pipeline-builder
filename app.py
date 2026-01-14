import streamlit as st
import pandas as pd

st.write("HELLO STREAMLIT")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

import data_ingestion
st.write("data_ingestion imported successfully")

import preprocessing
st.write("preprocessing imported successfully")

import model_trainer
st.write("model_trainer imported successfully")
