# K-Means Clustering

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, [3, 4]].values # Independent

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
# We will make a loop to go throught the 10 iterations
wcss = []
for i in range(1, 11):
    # First we will fit the Kmeans algorithm to out data X
    kmeans = KMeans(n_clusters = i, init ='k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    # Second we will compute the wcss (within-cluster sums of squares)  and append to our wcss[] 
    wcss.append(kmeans.inertia_)
    
# Plot
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()