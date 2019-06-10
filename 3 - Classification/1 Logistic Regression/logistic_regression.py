# Logistic Regression 

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# We separate the dependent and the independent variables
X = dataset.iloc[:, [2, 3]].values # Independent
y = dataset.iloc[:, 4].values # Dependent

# Taking care of missing data (giving missing data an average value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3]) # Just the columns with the missing values
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# test_size is the size of the data that will go to the test set. The rest will go to the training set. (0.25 = 25%)

# Feature Scaling (values in same scale for no dominant variable)
# test set no fit because it is already done in the training one
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)