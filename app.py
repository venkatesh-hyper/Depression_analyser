import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model and scaler from Hugging Face
scaler_path = hf_hub_download(repo_id="vengen9840/Depression_predict", filename="scaler.pkl")
model_path = hf_hub_download(repo_id="vengen9840/Depression_predict", filename="ensemble_model.pkl")

model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])

    # Mapping categorical values
    mapping = {'Yes': 1, 'No': 0}
    input_df[['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']] = \
        input_df[['Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']].replace(mapping)

    duration_mapping = {
        "Less than 5 hours": 4.0,
        "5-6 hours": 5.5,
        "6-7 hours": 6.5,
        "7-8 hours": 7.5,
        "8-9 hours": 8.5,
        "9-11 hours": 10.0,
        "More than 8 hours": 9.0,
    }
    input_df["sleep"] = input_df["Sleep Duration"].map(duration_mapping).fillna(0)

    # Gender Mapping
    gender_mapping = {"Male": 1, "Female": 0}
    input_df["Gender"] = input_df["Gender"].map(gender_mapping)

    # Combine fields
    input_df["Pressure"] = input_df["Academic Pressure"] + input_df["Work Pressure"]
    input_df["Satisfaction"] = input_df["Job Satisfaction"] + input_df["Study Satisfaction"]

    # Select features
    features = ['Gender', 'Age', 'sleep', 'Work/Study Hours', 'Pressure', 'Financial Stress', 'Satisfaction',
                'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    X_input = input_df[features].fillna(0)

    return scaler.transform(X_input)

st.title("Depression Prediction App")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "8-9 hours", "9-11 hours", "More than 8 hours"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0, value=8)
pressure = st.number_input("Academic/Work Pressure", min_value=0, value=2)
financial_stress = st.number_input("Financial Stress", min_value=0, max_value=10, value=0)
satisfaction = st.number_input("Job/Study Satisfaction", min_value=0, max_value=10, value=7)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])

user_input = {
    'Gender': gender,
    'Age': age,
    'Sleep Duration': sleep_duration,
    'Work/Study Hours': work_study_hours,
    'Academic Pressure': pressure,
    'Work Pressure': pressure,
    'Financial Stress': financial_stress,
    'Job Satisfaction': satisfaction,
    'Study Satisfaction': satisfaction,
    'Family History of Mental Illness': family_history,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts
}

if st.button("Predict"):
    X_input_scaled = preprocess_input(user_input)
    prediction = model.predict(X_input_scaled)[0]
    if prediction == 1:
        st.error("The model predicts you may have depression. Please consult with a mental health professional.")
    else:
        st.success("The model predicts you do not have depression.")
