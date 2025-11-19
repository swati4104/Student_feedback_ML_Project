import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------------------------------------
# LOAD MODEL + DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("New.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_data():
    return pd.read_csv("Employee_clean.csv")


model = load_model()
data = load_data()

st.title("🔮 Employee Prediction App")
st.write("Use this interface to make predictions using your trained ML model.")

# ---------------------------------------------------------
# AUTOMATIC FEATURE DETECTION
# ---------------------------------------------------------
try:
    feature_names = model.feature_names_in_
except:
    feature_names = data.columns.tolist()  # fallback


numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

# Only use columns the model expects
numeric_cols = [c for c in numeric_cols if c in feature_names]
categorical_cols = [c for c in categorical_cols if c in feature_names]

# ---------------------------------------------------------
# BUILD USE

