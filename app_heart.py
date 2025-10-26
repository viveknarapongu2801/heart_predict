import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('model_heart.pkl')
ct = joblib.load('encoder.pkl')
le_sex = joblib.load('le_sex.pkl')
le_exercise_angina = joblib.load('le_exercise_angina.pkl')

st.title("❤️ Heart Disease Prediction App")

st.write("""
This app predicts the likelihood of heart disease based on user inputs.
""")

# Collect inputs
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure', 50, 200, 120)
cholesterol = st.number_input('Cholesterol', 0, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Angina', ['N', 'Y'])
oldpeak = st.number_input('Oldpeak', 0.0, 6.2, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# Create dataframe
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

# Encode
input_df['Sex'] = le_sex.transform(input_df['Sex'])
input_df['ExerciseAngina'] = le_exercise_angina.transform(input_df['ExerciseAngina'])

# One-hot encode categorical columns using the same transformer as training
input_encoded = ct.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"🚨 High risk of Heart Disease (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of Heart Disease (Confidence: {prob:.2f})")
