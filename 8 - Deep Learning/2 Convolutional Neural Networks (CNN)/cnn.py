# Artificial Neural Network

# Installing Keras, Tensorflow and Theano
# conda install -c conda-forge keras

# Given that theere are only images will not have a data prepocessing part
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # To inicialize the Neural Network as a sequence of layers
from keras.layers import Convolution2D # To use in the first step; the Convolution Step. 2D because of images
from keras.layers import MaxPooling2D # Step 2: Pooling
from keras.layers import Flatten # Step 3: Flattening
from keras.layers import Dense # Add the fully connected layers and a classic ANN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution layer  
# Convolution2D: We create 32 feature detectors of 3x3 dimensions
# input_shape: format of the images.Size 64x64 and 3 because they are coloured ones  
# activation = 'relu' to get nonlinearity
classifier.add(Convolution2D(32, 3, 3, input_shaipe = (64, 64, 3), activation = 'relu'))