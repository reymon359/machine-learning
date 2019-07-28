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
# Convolution2D: We create 32 feature detectors of 3x3 dimensions. We will obtain 32 feature maps
# input_shape: format of the images.Size 64x64 and 3 because they are coloured ones  
# activation = 'relu' to get nonlinearity
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Applying Max Pooling to reduce the size of future maps and therefore reduce the number
# of nodes in future fully connected layers and improving performance
# pool_size = (2,2). With 2x2 we keep the information and we are still being precise
# on where we have the high numbers in the feature maps.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Putting all the numbers from the feature maps cells into one single vector, we still
# keep the feature maps high numbers in this vector which represent the spacial structure
# of the input image 
# Step 3 - Flattening
classifier.add(Flatten())