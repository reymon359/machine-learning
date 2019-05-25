# Multiple Linear Regression

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, :-1].values # Independent
y = dataset.iloc[:, 4].values # Dependent

# Encoding categorical data (not numbers)
# Encoding the Independent Variable (countries)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# Using Dummy encoding (splitting in to columns with same values 0 and 1)
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap 
X = X[:, 1:] # We use all dummy variables but 1
# Most of libraries already do it so it is not necessary

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# test_size is the size of the data that will go to the test set. The rest will go to the training set. (0.2 = 20%)

# Feature Scaling (values in same scale for no dominant variable)
# test set no fit because it is already done in the training one
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # We create a regressor object
regressor.fit(X_train, y_train) # We fit it in the training set

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Now we will use Backward Elimination to remove some not statically significant
# independent variables to get better predictions 😄😄
import statsmodels.formula.api as sm
# We need to add an extra independent variable to use the library (column of 50x1)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) 
# Now we will create the optimal Matrix with all the possible predictors
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  
# Now we will get for each independent variable its Pvalue that we will compare 
# to the significant level (0.05) to decide if we leave it or remove it from the model
# If it is greater we remove it
regresor_OLS.summary() # Seems like the x2 has the highest value (0.990) so we remove it and keep on

X_opt = X[:, [0, 1, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary() # Now it is the x1 (0.940)

X_opt = X[:, [0, 3, 4, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary() # Now the x2 (0.602) which refers to the 4 column

X_opt = X[:, [0, 3, 5]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary() # Now the x2 (0.060) which refers to the 5 column

X_opt = X[:, [0, 3]]
regresor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresor_OLS.summary() # Done, now none has a Pvalue above 0.05 so the most 
# significant variable in our model is the 3rd one.