# Upper Confidence Bound

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB (no external packages)
# Number of times ad i was selected up to round n
numbers_of_selections = [0] * d
# Sum of rewards of the ad i up to round n
sums_of_rewards = [0] * d


