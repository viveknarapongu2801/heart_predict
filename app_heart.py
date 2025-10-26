import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# 1️⃣ Load model and encoders
# -------------------------------
model = joblib.load('model_heart.pkl')
ct = joblib.load('encoder.pkl')
le_sex = joblib.load('le_sex.pkl')
le_exercise_angina = joblib.load('le_exercise_angina.pkl')

# -------------------------------
# 2️⃣ Streamlit App Title
# -------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts the **likelihood of heart disease** based on user inputs.  
Fill in the fields below and click **Predict**.
""")

# -------------------------------
# 3️⃣ User Inputs
# -------------------------------
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 50, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dL)', 0, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Max Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Angina', ['N', 'Y'])
oldpeak = st.number_input('Oldpeak', 0.0, 6.2, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# -------------------------------
# 4️⃣ Create DataFrame (each value wrapped in list)
# -------------------------------
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

# -------------------------------
# 5️⃣ Apply Encoders (same as training)
# -------------------------------
try:
    input_df['Sex'] = le_sex.transform(input_df['Sex'])
    input_df['ExerciseAngina'] = le_exercise_angina.transform(input_df['ExerciseAngina'])
except Exception as e:
    st.error(f"Encoder Error: {e}")
    st.stop()

# -------------------------------
# 6️⃣ One-Hot Encode using fitted ColumnTransformer
# -------------------------------
try:
    input_encoded = ct.transform(input_df)
except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# -------------------------------
# 7️⃣ Predict
# -------------------------------
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        if prediction == 1:
            st.error(f"🚨 High Risk of Heart Disease (Confidence: {probability:.2f})")
        else:
            st.success(f"✅ Low Risk of Heart Disease (Confidence: {probability:.2f})")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
