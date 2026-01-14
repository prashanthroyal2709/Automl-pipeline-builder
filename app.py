import streamlit as st
import pandas as pd

st.write("HELLO STREAMLIT")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("CSV loaded")
    st.dataframe(df.head())
