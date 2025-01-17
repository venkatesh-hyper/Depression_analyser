import pandas as pd
import numpy as np
import streamlit as st
import joblib
import gdown
import os

# Function to download the model from Google Drive
def download_model_from_drive():
    url = "https://drive.google.com/uc?id=1OSRR2QRy3sBrne9A30OqOkMZUDSeX9Ck&export=download"
    output = "ensemble_model.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return joblib.load(output)

# Load the trained model and scaler
model = download_model_from_drive()
scaler = joblib.load('scaler.pkl')  # Ensure that scaler.pkl is locally available

# Function to preprocess user input
def preprocess_input(user_input):
    """
    Processes and scales user input for prediction.
    """
    input_df = pd.DataFrame([user_input])

    # Map binary columns
    binary_map = {'Yes': 1, 'No': 0}
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(binary_map)
    input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map(binary_map)

    # Map sleep duration
    duration_mapping = {
        "Less than 5 hours": 4.0, "5-6 hours": 5.5, "6-7 hours": 6.5,
        "7-8 hours": 7.5, "8-9 hours": 8.5, "9-11 hours": 10.0,
        "More than 8 hours": 9.0
    }
    input_df['sleep'] = input_df['Sleep Duration'].map(duration_mapping).fillna(0)

    # Map gender
    input_df['Gender'] = input_df['Gender'].map({"Male": 1, "Female": 0})

    # Ensure numeric columns
    numeric_cols = ['Age', 'Work/Study Hours', 'Financial Stress', 'Job/Study Satisfaction', 'Academic/Work Pressure']
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Create additional features
    input_df['Pressure'] = input_df['Academic/Work Pressure']
    input_df['Satisfaction'] = input_df['Job/Study Satisfaction']

    # Select relevant features
    features = [
        'Gender', 'Age', 'sleep', 'Work/Study Hours', 'Pressure',
        'Financial Stress', 'Satisfaction', 'Family History of Mental Illness',
        'Have you ever had suicidal thoughts ?'
    ]
    X_input = input_df[features].fillna(0)

    # Scale the input
    X_scaled = scaler.transform(X_input)
    return X_scaled

# Streamlit app
def main():
    # App Title and Description
    st.title("Depression Prediction App")
    st.markdown(
        """
        This app uses a machine learning model to predict the likelihood of depression 
        based on your inputs. Please provide accurate information for the best results.
        """
    )

    # Layout: Two-column design for inputs
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
        age = st.number_input("Age", min_value=1, max_value=100, value=25, help="Enter your age.")
        sleep_duration = st.selectbox(
            "Sleep Duration", [
                "Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours",
                "8-9 hours", "9-11 hours", "More than 8 hours"
            ],
            help="Select your average sleep duration."
        )
        work_study_hours = st.number_input(
            "Work/Study Hours (per week)", min_value=0, value=40,
            help="Enter the total hours you spend working or studying per week."
        )
        financial_stress = st.slider(
            "Financial Stress (0-10)", 0, 10, 0,
            help="Rate your level of financial stress on a scale from 0 (none) to 10 (extreme)."
        )

    with col2:
        pressure = st.slider(
            "Academic/Work Pressure (0-10)", 0, 10, 2,
            help="Rate the level of academic or work-related pressure you feel."
        )
        satisfaction = st.slider(
            "Job/Study Satisfaction (0-10)", 0, 10, 7,
            help="Rate your satisfaction with your job or studies."
        )
        family_history = st.selectbox(
            "Family History of Mental Illness", ["Yes", "No"],
            help="Do you have a family history of mental illness?"
        )
        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts?", ["Yes", "No"],
            help="Have you experienced suicidal thoughts?"
        )

    # Collect user input
    user_input = {
        'Gender': gender,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Job/Study Satisfaction': satisfaction,
        'Academic/Work Pressure': pressure,
        'Family History of Mental Illness': family_history,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts
    }

    # Predict and display results
    if st.button("Predict"):
        try:
            X_input_scaled = preprocess_input(user_input)
            prediction = model.predict(X_input_scaled)[0]

            # Display prediction result
            st.markdown("<h3 style='text-align: center;'>Prediction Result:</h3>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(
                    "<p style='text-align: center; color:red;'>"
                    "The model predicts you may have depression. Please consult a mental health professional."
                    "</p>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<p style='text-align: center; color:green;'>"
                    "The model predicts you are unlikely to have depression."
                    "</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run the app
if __name__ == "__main__":
    main()
