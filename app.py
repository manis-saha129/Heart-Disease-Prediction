import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Split the data into features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)


# Define a function for prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]


# Streamlit app
st.title('Heart Disease Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

# Make prediction
if st.button('Predict'):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    result = heart_disease_prediction(input_data)
    if result == 0:
        st.success('The person does not have heart disease.')
    else:
        st.error('The person has heart disease.')

# Show model accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
st.write(f"Model Accuracy: {test_data_accuracy * 100:.2f}%")
