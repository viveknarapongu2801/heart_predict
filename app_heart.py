import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("model_heart.pkl")

# App title
st.title("❤️ Heart Disease Prediction App")

st.write("""
This app predicts the likelihood of **heart disease** based on your inputs.  
Please fill in the details below 👇
""")

# Input fields
age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.2, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Create dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [1 if sex == "M" else 0],
    "ChestPainType_ATA": [1 if chest_pain_type == "ATA" else 0],
    "ChestPainType_NAP": [1 if chest_pain_type == "NAP" else 0],
    "ChestPainType_ASY": [1 if chest_pain_type == "ASY" else 0],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "RestingECG_Normal": [1 if resting_ecg == "Normal" else 0],
    "RestingECG_ST": [1 if resting_ecg == "ST" else 0],
    "MaxHR": [max_hr],
    "ExerciseAngina": [1 if exercise_angina == "Y" else 0],
    "Oldpeak": [oldpeak],
    "ST_Slope_Up": [1 if st_slope == "Up" else 0],
    "ST_Slope_Flat": [1 if st_slope == "Flat" else 0],
})

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease (Confidence: {prob:.2f})")

st.caption("Developed with ❤️ using Streamlit and XGBoost")
