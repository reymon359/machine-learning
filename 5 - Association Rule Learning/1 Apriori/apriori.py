# Apriori

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
# Here python thought wrongly and set the first transaction names as headers
# So we put the header = None to tell him there are no headers in the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Now we will create a list of lists because is what we need, not a dataframe
transactions = []
for i in range(0, 7501): # Going through all the transactions
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Taining Apriori on the dataset
from apyori import apriori
# product bought 3 times a day = 3x7/7500 = 0.0028 = min_support
# min confidence of 0.2 goes well with the 0.003 min support
# For the min lift we are looking for rules with at least a 3
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)