# Upper Confidence Bound

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB (no external packages)
import math
N = 10000
d = 10 # Number of ads
ads_selected = [] # Ads selected on each iteration

# Step 1
# Number of times ad i was selected up to round n
numbers_of_selections = [0] * d
# Sum of rewards of the ad i up to round n
sums_of_rewards = [0] * d
total_reward = 0

# Step 2
# We need to compute for each version of the ad the 
# average reward and the confidence interval
for n in range(0, N):
    ad = 0 # Ad with the max_upper_bound 
    max_upper_bound = 0
    for i in range(0, d):
        # If the ad version i was selected at least once
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i 
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
    
    