# SVR Support Vector Regression

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
# The SVR class does not do it so we do it manually
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)
# Had this problem with the code upside "ValueError: Expected 2D array, got 1D array instead"
y = sc_y.fit_transform(y.reshape(-1,1))

# Fitting SVR to the dataset 
from sklearn.svm import SVR 
regressor = SVR(kernel = 'rbf') # rbf is the most common for non linear
regressor.fit(X, y)

# Predicting a new result 
# As we did feature scaling we will need to adapt this too
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the SVR  results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X),  color = 'blue') 
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary') 
plt.show()
  