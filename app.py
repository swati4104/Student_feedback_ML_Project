import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model and Data
# -----------------------------
@st.cache_resource
def load_model():
    with open("New.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv("Employee_clean.csv")

model = load_model()
data = load_data()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Employee Prediction App")

st.write("This app uses a trained ML model to make predictions based on employee data.")

# Example input fields (modify based on your model features)
st.sidebar.header("Input Features")

# Automatically detect numeric columns
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

user_input = {}

# Numeric inputs
for col in numeric_cols:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    default_val = float(data[col].median())
    user_input[col] = st.sidebar.number_input(
        f"{col} ({min_val} - {max_val})", 
        min_value=min_val, 
        max_value=max_val, 
        value=default_val
    )

# Categorical inputs
for col in categorical_cols:
    unique_vals = data[col].dropna().unique().tolist()
    user_input[col] = st.sidebar.selectbox(f"{col}", unique_vals)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

st.subheader("Your Input")
st.write(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        prediction = model.pre
