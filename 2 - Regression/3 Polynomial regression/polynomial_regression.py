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

# We will now create a Linear Regression and a Polynomial Regression model to
# compare them and see the differences between them
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # Creating the object
lin_reg.fit(X, y) # Fitting it

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures
# The greater the degree the better the results
poly_reg = PolynomialFeatures(degree = 4) # this will transform the matrix of 
# features X into a new one called poly with not just the variables but its exponentials too
X_poly = poly_reg.fit_transform(X) # The firs column its the constant

# Create a new Linear Regression model to not confuse with the first one and fit
# it with X_poly and y
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression)') 
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
# We now do not use X_poly because it was already defined for a matrix and we want a new one
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') 
plt.title('Truth or bluff (Polynomial Regression)') 
plt.xlabel('Position level')
plt.ylabel('Salary') 
plt.show()
# We can see a non linear model

# Increasing precission
X_grid = np.arange(min(X), max(X), 0.1 ) # To increase the plot resolution
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
# We now do not use X_poly because it was already defined for a matrix and we want a new one
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') 
plt.title('Truth or bluff (Polynomial Regression)') 
plt.xlabel('Position level')
plt.ylabel('Salary') 
plt.show()