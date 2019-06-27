# K-Means Clustering

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, [3, 4]].values # Independent