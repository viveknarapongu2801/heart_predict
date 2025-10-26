import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Load your trained model here
model = joblib.load('model_heart.pkl')

st.title('Heart Disease Prediction App')

st.write("""
This app predicts the likelihood of heart disease based on your inputs.
Please fill in the details below:
""")

# Define the features and their types (based on your preprocessing in the notebook)
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
binary_features = ["Sex", "ExerciseAngina", "FastingBS"]
categorical_features = ["ChestPainType", "RestingECG", "ST_Slope"]

# Create input fields for your model's features
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure (RestingBP)', 50, 200, 120)
cholesterol = st.number_input('Cholesterol', 0, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL (FastingBS)', [0, 1])
resting_ecg = st.selectbox('Resting Electrocardiogram (RestingECG)', ['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Maximum Heart Rate Achieved (MaxHR)', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina (ExerciseAngina)', ['N', 'Y'])
oldpeak = st.number_input('Oldpeak', 0.0, 6.2, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])


# Create a dictionary with user inputs
user_input = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain_type,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': resting_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope
}

# Convert user input to a pandas DataFrame
input_df = pd.DataFrame([user_input])

# Preprocess the input data
# You need to replicate the preprocessing steps from your notebook
# Label encode binary features
le_sex = LabelEncoder()
input_df['Sex'] = le_sex.fit_transform(input_df['Sex']) # Fit and transform on a small sample that includes both 'M' and 'F' if possible to avoid errors

le_exercise_angina = LabelEncoder()
input_df['ExerciseAngina'] = le_exercise_angina.fit_transform(input_df['ExerciseAngina']) # Fit and transform on a small sample that includes both 'N' and 'Y' if possible to avoid errors

# One-hot encode categorical features
# To ensure consistent columns, fit on the original training data features if possible
# For a simple app, we can define the transformer with the known categories
ct = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop='first', sparse_output=False, categories=[['ATA', 'NAP', 'ASY', 'TA'], ['Normal', 'ST', 'LVH'], ['Up', 'Flat', 'Down']]), categorical_features)
    ],
    remainder='passthrough'  # keep other columns
)

# Create a dummy DataFrame with all possible categories to fit the ColumnTransformer
# This is a workaround to ensure the transformer has all categories even if the user input doesn't include them all
dummy_data = pd.DataFrame({
    'Sex': ['M', 'F'],
    'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Up', 'Flat', 'Down'],
    'Age': [50, 50],
    'RestingBP': [120, 120],
    'Cholesterol': [200, 200],
    'FastingBS': [0, 1],
    'MaxHR': [150, 150],
    'Oldpeak': [1.0, 1.0]
})

# Apply the same label encoding to dummy data for fitting the ColumnTransformer
le_sex_dummy = LabelEncoder()
dummy_data['Sex'] = le_sex_dummy.fit_transform(dummy_data['Sex'])

le_exercise_angina_dummy = LabelEncoder()
dummy_data['ExerciseAngina'] = le_exercise_angina_dummy.fit_transform(dummy_data['ExerciseAngina'])


ct.fit(dummy_data[categorical_features + numeric_features + binary_features]) # Fit ColumnTransformer with all features

input_encoded = ct.transform(input_df)

# Convert the numpy array back to a DataFrame with correct column names if needed for your model (XGBoost usually works with numpy)
# ohe_cols = ct.named_transformers_["onehot"].get_feature_names_out(categorical_features)
# encoded_col_names = list(ohe_cols) + [col for col in input_df.columns if col not in categorical_features]
# input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_col_names)


# Make prediction when a button is clicked
if st.button('Predict'):
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)[:, 1] # For probability

    # Display the prediction result
    if prediction[0] == 1:
        st.error('Prediction: High risk of heart disease')
    else:
        st.success('Prediction: Low risk of heart disease')

    st.write(f'Confidence: {prediction_proba[0]:.2f}')

st.write("Please fill in all the required fields and click 'Predict'.")
