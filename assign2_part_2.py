# Group B: Assignment 2 (Part 2): Data Analytics using Python <br>
# Perform the following operations using Python on Heart Diseases data sets <br>
# # Dataset ( Heart Diseases ) : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# e. Data model building 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Processing
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/home/mangal/Downloads/heart.csv') # put her path of your dataset


# print first 5 rows of the dataset
heart_data.head()


# print last 5 rows of the dataset
heart_data.tail()


# number of rows and columns in the dataset
heart_data.shape


# info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of "target" variable
heart_data['target'].value_counts()


# 1 --> Defective Heart
# 
# 0 --> Healthy Heart
# 
# Splitting the Features and Target

X= heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

print(X)

print(Y)


# Splitting the Data into Training Data and Test Data


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


print(X.shape, X_train.shape, X_test.shape)


print(X_test.head())


# Model Training
# 
# Logistic Regression


model = LogisticRegression(max_iter=1050)


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# Model Evaluation: finding accuracy on training data

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracyy on training data : ', training_data_accuracy)


# accuracy on test data
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

type(X_test)

print(X_test_prediction)

# Building a Predictive System

input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3) # single instance of 13 features

# change the input data to a numpy array

input_data_as_numpy_array= np.array(input_data)

# reshape the numpy array as we are predicting for only on instance
# so we want to reshape it to 1*13 : 1 sample 13 features
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

test_df = pd.DataFrame(input_data_reshaped, columns = X_test.columns )

prediction = model.predict(test_df)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

