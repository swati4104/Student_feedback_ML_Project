import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Constants and Setup ---
# The model file path and expected feature names
MODEL_FILE = 'New.pkl'
# The features are inferred from the column names in Employee_clean.csv (excluding 'Feedback')
FEATURE_NAMES = ['Age', 'Experience', 'Salary', 'Department'] 
# Mapping for Department IDs (inferred from common data practices, adjust if your data uses different labels)
DEPARTMENT_MAP = {
    0: 'Sales',
    1: 'IT',
    2: 'HR',
    3: 'Finance',
    4: 'R&D'
}

# --- Utility Functions ---

@st.cache_resource
def load_model():
    """Loads the pickled model file."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' not found.")
        st.stop()
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def make_prediction(model, data_df):
    """
    Makes a prediction using the loaded model.
    The model is expected to be a scikit-learn compatible classifier.
    """
    try:
        # Ensure the input DataFrame columns match the model's expected features
        X = data_df[FEATURE_NAMES]
        prediction = model.predict(X)
        # The model is likely predicting the class labels ('Good', 'Average', 'Poor') directly
        return prediction[0] 
    except Exception as e:
        st.error(f"Prediction failed. Check model compatibility and input data types/format. Error: {e}")
        return "Prediction Error"

# --- Streamlit App Layout ---

def main():
    st.set_page_config(page_title="Employee Feedback Prediction App", layout="centered")

    # Load the model
    model = load_model()

    # Title and Description
    st.title("💡 Employee Feedback Prediction")
    st.markdown("""
        Enter the employee details below to predict their likely 'Feedback' category (e.g., Good, Average, Poor) 
        based on the trained machine learning model (`New.pkl`).
    """)

    # --- User Input Sidebar/Section ---

    st.header("Employee Data Input")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (Years)", min_value=18.0, max_value=65.0, value=30.0, step=1.0)
    
    with col2:
        experience = st.slider("Experience (Years)", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

    # Convert dictionary keys to list for the selectbox options
    department_options = list(DEPARTMENT_MAP.values())
    department_label = st.selectbox(
        "Department",
        options=department_options
    )
    # Get the Department ID (key) from the label (value)
    # The model expects the numerical encoded value (0, 1, 2, 3, 4)
    department_id = [k for k, v in DEPARTMENT_MAP.items() if v == department_label][0]


    salary = st.number_input(
        "Salary ($)", 
        min_value=10000, 
        max_value=200000, 
        value=50000, 
        step=1000,
        format="%d"
    )

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Salary': [salary],
        'Department': [department_id] # Use the numerical ID
    })

    # --- Prediction Button ---

    if st.button("Predict Feedback", type="primary"):
        # Display a spinner while predicting
        with st.spinner('Predicting...'):
            prediction = make_prediction(model, input_data)

            # Display Results
            st.markdown("---")
            st.subheader("Prediction Result")
            
            # Simple color coding for the output
            if 'Good' in str(prediction):
                color = 'green'
                icon = '⭐'
            elif 'Poor' in str(prediction):
                color = 'red'
                icon = '❌'
            else:
                color = 'orange'
                icon = '🟡'

            st.markdown(
                f"The predicted **Employee Feedback** is: <span style='color:{color}; font-size: 24px;'>{icon} **{prediction}**</span>",
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            st.caption("Input Data Used for Prediction:")
            st.dataframe(input_data)


if __name__ == "__main__":
    main()
