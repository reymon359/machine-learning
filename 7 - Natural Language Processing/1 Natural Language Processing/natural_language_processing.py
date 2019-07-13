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
import nltk # Library for the irrelevant words
nltk.download('stopwords') # list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # for Stemming
corpus = [] # The corpus will be the collection with all the cleaned text
for i in range(0, 1000): 
    # First param is what we don't want to remove
    # Second param is what to put in the removed character. We will put a space
    # Third param is what we want to clean
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    # Now we will put all the letters into lowecase
    review = review.lower()
    # Now we will remove irrelevant words like prepositions
    # First we will split all the text into a list of words
    review = review.split()
    # Now that the review is a list of words we will go through it and remove the
    # irrelevant ones according to the nltk stopwords. We have to specify that it
    # is in english. We will also put the list in a set to faster the process.
    # We will also apply Stemming which is about taking the route of the words. 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('English'))]
    # Now we will join the words back
    review = ' '.join(review)
    corpus.append(review)
    
# -- Creating the Bag of Words model --
# It will be like the classification model and will predict if the review is
# positive or negative
from sklearn.feature_extraction.text import CountVectorizer
# The for loop done before can be done here too with some params
cv = CountVectorizer(max_features = 1500) # Reducing the sparsity
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values # Dependant variable for model 

# -- Applying Naive Bayes --
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# test_size is the size of the data that will go to the test set. The rest will go to the training set. (0.25 = 25%)

# Fitting the classifier to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() # No arguments, very simple
classifier.fit(X_train, y_train)

# Predicting the Test set results
# y_pred = vector of predictions of each of the test set observations.
y_pred = classifier.predict(X_test)

# Making the confusion matrix
# Now we will evaluate the predictions
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

# Accuracy
(55 + 91)/200
