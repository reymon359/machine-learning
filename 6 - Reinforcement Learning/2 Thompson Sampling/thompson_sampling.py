# Thompson Sampling

# Importing the libraries
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling (no external packages)
import random
N = 10000
d = 10 # Number of ads
ads_selected = [] # Ads selected on each iteration

# Step 1
# For each ad the number of times they got reward 1 or 0
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

# Step 2
# We need to compute for each version of the ad the 
# average reward and the confidence interval
for n in range(0, N):
    ad = 0 # Ad with the max_upper_bound 
    max_random = 0
    for i in range(0, d):
        # Here we will implement the algorithm
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)        
        if random_beta > max_random:
            max_random = random_beta
            ad = i 
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        # We implement the number of rewards of that ad
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    total_reward = total_reward + reward
    
# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
    
    


