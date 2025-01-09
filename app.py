import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model and scaler from Hugging Face
scaler_path = hf_hub_download(repo_id="vengen9840/Depression_predict", filename="scaler.pkl")
model_path = hf_hub_download(repo_id="vengen9840/Depression_predict", filename="ensemble_model.pkl")

# Load the trained model and scaler
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Function to preprocess user input
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    # Mapping and preprocessing
    mapping = {'Yes': 1, 'No': 0}
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(mapping)
    input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map(mapping)
    
    duration_mapping = {
        "Less than 5 hours": 4.5, "5-6 hours": 5.5, "6-7 hours": 6.5, 
        "7-8 hours": 7.5, "8-9 hours": 8.5, "9-11 hours": 10.0, 
        "More than 8 hours": 9.0
    }
    input_df["Sleep Duration"] = input_df["Sleep Duration"].map(duration_mapping)

    input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})
    features = ['Gender', 'Age', 'Sleep Duration', 'Work/Study Hours', 
                'Financial Stress', 'Satisfaction', 'Family History of Mental Illness', 
                'Have you ever had suicidal thoughts ?']
    return scaler.transform(input_df[features])

# Streamlit app
st.title("Depression Prediction App")
st.write("Please fill in the details below to predict if you may have depression.")

# User Input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sleep_duration = st.selectbox("Sleep Duration", 
                               ["Less than 5 hours", "5-6 hours", "6-7 hours", 
                                "7-8 hours", "8-9 hours", "9-11 hours", "More than 8 hours"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0, value=8)
financial_stress = st.number_input("Financial Stress", min_value=0, max_value=10, value=0)
satisfaction = st.number_input("Job/Study Satisfaction", min_value=0, max_value=10, value=7)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])

user_input = {
    'Gender': gender, 'Age': age, 'Sleep Duration': sleep_duration,
    'Work/Study Hours': work_study_hours, 'Financial Stress': financial_stress,
    'Satisfaction': satisfaction, 
    'Family History of Mental Illness': family_history,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts
}

# Predict
if st.button("Predict"):
    X_input = preprocess_input(user_input)
    prediction = model.predict(X_input)[0]
    if prediction == 1:
        st.error("The model predicts you may have depression. Please consult a mental health professional.")
    else:
        st.success("The model predicts you do not have depression.")
