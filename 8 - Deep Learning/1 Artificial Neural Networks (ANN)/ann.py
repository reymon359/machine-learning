# Artificial Neural Network

# Installing Keras, Tensorflow and Theano
# conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, 3:13].values # Independent
y = dataset.iloc[:, 13].values # Dependent

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



