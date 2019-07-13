# Natural Language Processing

# -- Importing the libraries --
import numpy as np #  To work with mathematical numbers.
import matplotlib.pyplot as plt #  To work with plots
import pandas as pd # To import and manage datasets
 
# --  Importing the dataset -- 
# We will use a .tsv file over a .csv because the texts already have commas 
# and we can not separate by them in csv because we will obtain a wrong dataset
# delimiter = '\t' to separate by tabs 
# quoting = 3 to ignore double quote
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# -- Cleaning the text --
# We will use a library for it
import re
# First param is what we don't want to remove
# Second param is what to put in the removed character. We will put a space
# Third param is what we want to clean
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) 