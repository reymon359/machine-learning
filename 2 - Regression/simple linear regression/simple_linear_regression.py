# Simple Linear Regression

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, :-1].values # Independent
y = dataset.iloc[:, 1].values # Dependent

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling (values in same scale for no dominant variable)
# test set no fit because it is already done in the training one
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
# We will no t need it for this so I left it commented

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Creating an object of that class
regressor.fit(X_train, y_train) # Fits the regressor object to the trainning set

# Now our Simple Linear Regressor has 'learnt' the correlations and can predict
# Predicting the Test set results.
y_pred  =  vector of predictions of the dependent variable