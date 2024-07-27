# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing
# Loading the csv data to a pandas dataframe
heart_data = pd.read_csv('heart_disease_data.csv')

# Print first five rows of the dataset
print(heart_data.head())
# Print last five rows of the dataset
print(heart_data.tail())
# Number of rows and column in the dataset
print(heart_data.shape)
# Getting some info about the data
print(heart_data.info())
# Checking for missing values
print(heart_data.isnull().sum())
# Statistical measures about the data
print(heart_data.describe())
# Checking the distribution of target variable
print(heart_data['target'].value_counts())
# 1 --> Defective Heart
# 0 --> Healthy Heart
# Splitting the features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Logistic Regression Model
model = LogisticRegression()
# Training the Logistic Regression Model with training data
model.fit(X_train, Y_train)

# Model Evaluation - Accuracy Score
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data: ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data: ', test_data_accuracy)

# Building a Predictive System
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has a Heart Disease')
