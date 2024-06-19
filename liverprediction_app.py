import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the model and preprocessing objects
model = joblib.load('liver_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the app description
st.title('Liver Disease Prediction')
st.subheader('App by Jonathan Ibifubara Pollyn')
st.write("Liver disease includes a wide range of conditions that can harm the liver, from mild swelling to severe conditions like cirrhosis and liver cancer. The liver is a vital organ that removes harmful substances, makes bile, stores essential nutrients, and keeps the metabolism in check. According to World Health Organization, 2021, an early detection of liver disease is crucial, as it can halt its progression, enhance treatment effectiveness, reduce complications, and improve patients' overall well-being. . This application delves into the development of a machine-learning model that can predict liver disease, discussing its execution, effectiveness, and potential impact, which could revolutionize the field of liver disease diagnosis. If you have any question about the application, you can contact me via email at j.pollyn@gmail.com")
st.markdown(
    """
        ## Attribute Information

- Age: Range: 20 to 80 years.
- Gender: Male (0) or Female (1).
- BMI (Body Mass Index): Range: 15 to 40.
- Alcohol Consumption: Range: 0 to 20 units per week.
- Smoking: No (0) or Yes (1).
- Genetic Risk: Low (0), Medium (1), High (2).
- Physical Activity: Range: 0 to 10 hours per week.
- Diabetes: No (0) or Yes (1).
- Hypertension: No (0) or Yes (1).
- Liver Function Test: Range: 20 to 100.
- Diagnosis: Binary indicator (0 or 1) of liver disease presence.

""")

# Define mapping for user friendly inputs
gender_mapping = {'Male': 0, 'Female':1}
smoking_mapping = {'No': 0, 'Yes': 1}
diabetes_mapping = {'No': 0, 'Yes': 1}
hypertension_mapping = {'No': 0, 'Yes': 1}
genetic_mapping = {'Low': 0, 'Medium': 1, 'High': 2}



# Collect user input
st.sidebar.header('User Input Parameters')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
bmi = st.sidebar.number_input('BMI (kg/m²)', min_value=0.0, max_value=1000.0, value=5.0)
alcohol_consumption = st.sidebar.number_input('Alcohol Consumption (unit Per Week)', min_value=0, max_value=20, value=0)
smoking_status = st.sidebar.selectbox('Smoking Status', ['No', 'Yes'])
genetic_risk = st.sidebar.selectbox('Genetic Risk?', ['Low', 'Medium', 'High'])
physical_activity = st.sidebar.number_input('Physical Activity (Hours Per Week)', min_value=0.0, max_value=1000.0, value=0.0)
diabetes = st.sidebar.selectbox('Diabetes?', ['No', 'Yes'])
hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
liver_function_test = st.sidebar.number_input('Liver Function Test', min_value=20.0, max_value=100.0, value=20.0)
age_range = st.sidebar.selectbox('Age Range', ['19-35', '36-50', '51-65', '66-80', '80+'])

# Map the user friendly to the user input
gender = gender_mapping[gender]
smoking_status = smoking_mapping[smoking_status]
diabetes = diabetes_mapping[diabetes]
hypertension = hypertension_mapping[hypertension]
genetic_risk = genetic_mapping[genetic_risk]


# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Gender': [gender],
    'BMI': [bmi],
    'AlcoholConsumption': [alcohol_consumption],
    'Smoking': [smoking_status],
    'GeneticRisk': [genetic_risk],
    'PhysicalActivity': [physical_activity],
    'Diabetes': [diabetes],
    'Hypertension': [hypertension],
    'LiverFunctionTest': [liver_function_test],
    'age_range': [age_range]
})

# Apply the label encoding
for column, le in label_encoders.items():
    input_data[column] = le.transform(input_data[column])

# Define columns to be scaled
columns_to_scale = ['Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest', 'age_range']

# Apply MinMaxScaler
input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

# Predict the new data
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.sidebar.write('The model predicts that there is a presence of liver disease in the patient liver.')
    else:
        st.sidebar.write('The model predicts that there are no presence of liver disease in the patient liver.')

