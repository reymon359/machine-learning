# Hierarchical Clustering

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values 

# Using the Dendrogram to fing the optimal number of clusters
import scipy.cluster.hierarchy as sch
# We now select the data and the method for the dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()