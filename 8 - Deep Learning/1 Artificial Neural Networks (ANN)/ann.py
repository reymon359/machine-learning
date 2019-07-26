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

