# Polynomial Regression
# We will build a ML model to predict the 6.5 number of years/salary relation
# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, 1:2].values # The 1:2 is to obtain a matrix not a vector
y = dataset.iloc[:, 2].values # Dependent

# We will not split the dataset for 2 reasons
# 1 The dataset is too small
# 2 We need to be as acurate as possible so we let it stay as it is
# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
# test_size is the size of the data that will go to the test set. The rest will go to the training set. (0.2 = 20%)

# Feature Scaling (values in same scale for no dominant variable)
# test set no fit because it is already done in the training one
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""